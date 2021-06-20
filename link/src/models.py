import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_sparse import spspmm
from torch_scatter import scatter
from torch.nn import ModuleList, Linear, Conv1d, MaxPool1d, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn import (GAE, GCNConv, LEConv, GraphConv, JumpingKnowledge, TopKPooling, ASAPooling,
                                global_add_pool, global_sort_pool, global_mean_pool, SAGPooling)
from torch_geometric.utils import (negative_sampling, remove_self_loops, sort_edge_index,
                                   dropout_adj, add_self_loops, dropout_adj)

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform

EPS = 1e-15
MAX_LOGSTD = 10

class KFAC(torch.optim.Optimizer):

    def __init__(self, net, eps=0.01, sua=False, pi=False, update_freq=1,
                 alpha=1.0, constraint_norm=False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        
        for mod in net.modules():
            mod_name = mod.__class__.__name__
            if mod_name in ['CRD', 'CLS']:
                handle = mod.register_forward_pre_hook(self._save_input)
                self._fwd_handles.append(handle)
                
                for sub_mod in mod.modules():
                    i_sub_mod = 0
                    if hasattr(sub_mod, 'weight'):
                        assert i_sub_mod == 0
                        handle = sub_mod.register_backward_hook(self._save_grad_output)
                        self._bwd_handles.append(handle)
                        
                        params = [sub_mod.weight]
                        if sub_mod.bias is not None:
                            params.append(sub_mod.bias)

                        d = {'params': params, 'mod': mod, 'sub_mod': sub_mod}
                        self.params.append(d)
                        i_sub_mod += 1

        super(KFAC, self).__init__(self.params, {})

    def step(self, update_stats=True, update_params=True, lam=0.):
        """Performs one step of preconditioning."""
        self.lam = lam
        fisher_norm = 0.
        for group in self.param_groups:
            
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group, state)
                    ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],
                                                state['num_locations'])
                    state['ixxt'] = ixxt
                    state['iggt'] = iggt
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state)

            if update_params:
                gw, gb = self._precond(weight, bias, group, state)

                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()

                weight.grad.data = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad.data = gb
                    
            # Cleaning
            if 'x' in self.state[group['mod']]:
                del self.state[group['mod']]['x']
            if 'gy' in self.state[group['mod']]:
                del self.state[group['mod']]['gy']
        
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group['params']:
                    print(param.shape, param)
                    param.grad.data *= scale

        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        # i = (x, edge_index)
        if mod.training:
            self.state[mod]['x'] = i[0]
            
            self.mask = i[-1]
            
    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(1)
            self._cached_edge_index = mod._cached_edge_index

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        ixxt = state['ixxt'] # [d_in x d_in]
        iggt = state['iggt'] # [d_out x d_out]
        g = weight.grad.data # [d_in x d_out]
        s = g.shape

        g = g.contiguous().view(-1, g.shape[-1])
            
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(1, gb.shape[0])], dim=0)

        g = torch.mm(ixxt, torch.mm(g, iggt))
        if bias is not None:
            gb = g[-1].contiguous().view(*bias.shape)
            g = g[:-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        sub_mod = group['sub_mod']
        x = self.state[group['mod']]['x'] # [n x d_in]
        gy = self.state[group['sub_mod']]['gy'] # [n x d_out]
        edge_index, edge_weight = self._cached_edge_index # [2, n_edges], [n_edges]
        
        n = float(self.mask.sum() + self.lam*((~self.mask).sum()))

        x = scatter(x[edge_index[0]]*edge_weight[:, None], edge_index[1], dim=0)
        
        x = x.data.t()

        if sub_mod.weight.ndim == 3:
            x = x.repeat(sub_mod.weight.shape[0], 1)
        


        if sub_mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)

        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / n
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / n)
        
        gy = gy.data.t() # [d_out x n]

        state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / n 
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / n)

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()

        return ixxt, iggt

    def __del__(self):
        for handle in self._fwd_handles + self._bwd_handles:
            handle.remove()

class SVGAE(GAE):
    
    def __init__(self, z_dim, encoder, decoder=None):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(SVGAE, self).__init__(encoder, decoder)
        self.z_dim = torch.LongTensor([z_dim]).squeeze().cuda()

    def encode(self, *args, **kwargs):
        self.__mu__, self.__std__ = self.encoder(*args, **kwargs)
        q_z, p_z = self.reparameterize(self.__mu__, self.__std__)
        
        return q_z.rsample()
        
    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)
    
    def kl_loss(self, mu=None, std=None):
        mu = self.__mu__ if mu is None else mu
        std = self.__std__ if std is None else std
        q_z, p_z = self.reparameterize(mu, std)
        
        return torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
        
    def reparameterize(self, mu=None, std=None):
        # print(mu.size(), logstd.size())
        q_z = VonMisesFisher(mu, std, validate_args=False)
        p_z = HypersphericalUniform(self.z_dim - 1, device='cuda', validate_args=False)

        return q_z, p_z
    
    
class DGCNN(torch.nn.Module):
    def __init__(self, train_dataset, hidden_channels, num_layers, GNN=GCNConv, k=0.6):
        super(DGCNN, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        if GNN == GCNConv:
            self.convs = ModuleList()
            self.convs.append(GNN(train_dataset.num_features, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(GNN(hidden_channels, hidden_channels))
            self.convs.append(GNN(hidden_channels, 1))
        else:
            self.convs = ModuleList()
            self.convs.append(GNN(train_dataset.num_features, hidden_channels, 3))
            for i in range(0, num_layers - 1):
                self.convs.append(GNN(hidden_channels, hidden_channels, 3))
            self.convs.append(GNN(hidden_channels, 1, 3))
            

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        # self.pooling = SAGPooling(total_latent_dim, ratio=self.k, GNN=GCNConv)
        
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, x, edge_index, batch):
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        # x = self.pooling(x, edge_index, batch=batch)[0]
        
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    
class SAGPoolNet(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=1.0):
        super(SAGPoolNet, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden, aggr='mean')
        self.convs = torch.nn.ModuleList()
        # self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden, aggr='mean')
            for i in range(num_layers - 1)
        ])
        # self.pools.extend(
        #     [SAGPooling(hidden, ratio) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')
        self.pooling = SAGPooling(num_layers * hidden, ratio)
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        # for pool in self.pools:
        #     pool.reset_parameters()
        self.pooling.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        # xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [x]
            # xs += [global_mean_pool(x, batch)]
            # if i % 2 == 0 and i < len(self.convs) - 1:
            #     pool = self.pools[i // 2]
            #     x, edge_index, _, batch, _, _ = pool(x, edge_index,
            #                                          batch=batch)
        x = torch.cat(xs[1:], dim=-1)
        x, _, _, batch, _, _ = self.pooling(x, edge_index, batch=batch)
        x = self.jump(x)
        # print(x.size())
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
    
class UNet(torch.nn.Module):
    def __init__(self, in_channels, hidden, num_layers,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden
        self.depth = num_layers
        self.pool_ratios = pool_ratios
        self.act = act
        self.sum_res = sum_res

        channels = hidden

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(num_layers):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))
            
        self.lin1 = Linear(channels, 128)
        self.lin2 = Linear(128, 1)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()

    def forward(self, x, edge_index, batch):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)
            
        x = self.act(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
            
        return x
    
    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

    def __repr__(self):
        return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.pool_ratios)
    
    
class ASAP(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, dropout=0):
        super(ASAP, self).__init__()
        # self.conv1 = GraphConv(dataset.num_features, hidden, aggr='mean')
        self.conv1 = LEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        # self.convs.extend([
        #     GraphConv(hidden, hidden, aggr='mean')
        #     for i in range(num_layers - 1)
        # ])
        self.convs.extend([
            LEConv(hidden, hidden)
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio, dropout=dropout)
            for i in range((num_layers) // 2)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, edge_weight, batch, _ = pool(
                    x=x, edge_index=edge_index, edge_weight=edge_weight,
                    batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
    
    
class ASAPDefault(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, ratio=0.8, dropout=0, dropedge=0.5):
        super(ASAPDefault, self).__init__()
        if type(ratio) != list:
            ratio = [ratio for i in range(num_layers)]
        self.dropedge = dropedge
            
        self.conv1 = GraphConv(dataset.num_features, hidden)
        self.pool1 = ASAPooling(hidden, ratio[0], dropout=dropout)
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden, hidden)
            for i in range(num_layers - 1)
        ])
        self.pools.extend([
            ASAPooling(hidden, ratio[i + 1], dropout=dropout)
            for i in range(num_layers - 1)
        ])
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = Linear(num_layers * hidden, hidden)
        self.lin2 = Linear(hidden, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        new_edge_index, new_edge_weight = dropout_adj(
            edge_index, edge_weight, self.dropedge, training=self.training)
        x = F.relu(self.conv1(x, new_edge_index, new_edge_weight))
        x, edge_index, edge_weight, batch, _ = self.pool1(
            x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        xs = [global_mean_pool(x, batch)]
        
        for i, conv in enumerate(self.convs):
            new_edge_index, new_edge_weight = dropout_adj(
                edge_index, edge_weight, self.dropedge, training=self.training)
            x = F.relu(conv(x=x, edge_index=new_edge_index, edge_weight=new_edge_weight))
            x, edge_index, edge_weight, batch, _ = self.pools[i](
                x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
            xs += [global_mean_pool(x, batch)]
        x = self.jump(xs)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
