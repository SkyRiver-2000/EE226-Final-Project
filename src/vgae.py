import time
import os.path as osp

import argparse
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

from src.utils import *
from src.seal_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--epochs', type=int, default=400)
args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj, features, labels, idx_train, idx_val, idx_test, loss_coef = load_data()

edge_list, edge_weight, edge_type = load_edges()

num_relations = np.shape(pd.unique(edge_type[:, 0]))[0]

edge_list, edge_weight, edge_type = torch.from_numpy(edge_list.T).long(), \
    torch.from_numpy(edge_weight).float(), torch.from_numpy(edge_type).long()
    
data = Data(x = features, edge_index = edge_list, edge_type = edge_type)

data = train_test_split_edges(data)
edge_index, _ = add_self_loops(data.train_pos_edge_index)

data.train_neg_edge_index = negative_sampling(
    edge_index, num_nodes=data.num_nodes,
    num_neg_samples=data.train_pos_edge_index.size(1)
)

data.val_neg_edge_index = negative_sampling(
    edge_index, num_nodes=data.num_nodes,
    num_neg_samples=data.val_pos_edge_index.size(1)
)

data.test_neg_edge_index = negative_sampling(
    edge_index, num_nodes=data.num_nodes,
    num_neg_samples=data.test_pos_edge_index.size(1)
)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


out_channels = 64
num_features = data.x.size(1)

if not args.variational:
    if not args.linear:
        model = GAE(GCNEncoder(num_features, out_channels))
    else:
        model = GAE(LinearEncoder(num_features, out_channels))
else:
    if args.linear:
        model = VGAE(VariationalLinearEncoder(num_features, out_channels))
    else:
        model = VGAE(VariationalGCNEncoder(num_features, out_channels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.variational:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def get_test_output():
    pairs_to_pred = [[], []]
    for pair in pd.read_csv("dataset/author_pairs_to_pred_with_index.csv")["author_pair"].values:
        temp = pair.split(' ')
        pairs_to_pred[0].append(int(temp[0]))
        pairs_to_pred[1].append(int(temp[1]))
        
    edge_index = (torch.LongTensor(pairs_to_pred) + 24251).to(device)
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
        pred = model.decode(z, edge_index, sigmoid=True)
        
    return pred.view(pred.size(0), 1).cpu().numpy()
        
best_auc = 0

for epoch in range(1, args.epochs + 1):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    if auc > best_auc:
        best_auc = auc
        test_output = get_test_output()
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

now_time = time.strftime("%m%d%H%M%S", time.localtime(int(time.time())))
pd.DataFrame(
  np.hstack([np.arange(test_output.shape[0]).reshape((test_output.shape[0], 1)), test_output]),
  columns=["id", "label"]
).to_csv("submission/{}.csv".format(now_time), index=False)
