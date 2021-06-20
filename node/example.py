import math
import time
import argparse
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from copy import deepcopy, copy

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
# from xgboost import XGBClassifier, XGBRFClassifier

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, ModuleList, Conv1d, MaxPool1d
from torch_geometric.utils import dropout_adj, k_hop_subgraph
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv, SGConv, GCNConv, GATConv, SuperGATConv, RGCNConv, GraphUNet

from src.loss import FocalLoss, F1_Loss
from src.utils import load_data, load_reference_edges, load_edges, get_paper_label, create_csv_from_result, filter_csv, compute_f1_score

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--conv", type=str, default="SG")
parser.add_argument('--epochs', type=int, default=1000,
                    help="Number of epochs to train.")
parser.add_argument('--lr', type=float, default=0.01,
                    help="Initial learning rate.")
parser.add_argument('--weight_decay', type=float, default=0,
                    help="Weight decay (L2 loss on parameters).")
parser.add_argument('--head', type=int, default=1,
                    help="Number of attention heads.")
parser.add_argument('--concat', action='store_true', default=False,
                    help="Concat outputs of all convolution layers.")
parser.add_argument('--training', action='store_true', default=True,
                    help="Whether to add validation labels as feature.")
parser.add_argument('--weighted_loss', action='store_true', default=False,
                    help="Use the scale of each class to reweight loss.")
parser.add_argument('--load_model', type=str, default="")
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj, features, labels, idx_train, idx_val, idx_test, loss_coef = load_data(training=args.training)

edge_list, edge_weight, edge_type = load_edges()
num_relations = np.shape(pd.unique(edge_type[:, 0]))[0]
edge_list, edge_weight, edge_type = torch.from_numpy(edge_list.T).long().cuda(), \
    torch.from_numpy(edge_weight).float().cuda(), torch.from_numpy(edge_type).long().cuda()
data = Data(x = features, y = labels, edge_index = edge_list,
            edge_type = edge_type, edge_weight = edge_weight).to(device)

n_epochs = args.epochs
conv_type = args.conv
if args.conv in ["GAT", "SuperGAT"]:
    n_attn_head = args.head
loss_coef = loss_coef.cuda()
now_time = time.strftime("%m%d%H%M%S", time.localtime(int(time.time())))

class Net(torch.nn.Module):
    def __init__(self,
                 num_features = 128,
                 hidden_units = 256,
                 num_layers = 1,
                 num_classes = 10,
                 dropout_rate = 0.5,
                 dropedge_rate = 0.75,
                 conv_type = None,
                 concat = False,
                 **kwargs
                 ):
        super(Net, self).__init__()
        
        dim = hidden_units
        self.concat = concat
        self.conv_type = conv_type
        self.dropout_rate = dropout_rate
        self.dropedge_rate = dropedge_rate
        
        # self.feature_encoder = Sequential(Linear(num_features, dim),
        #                                   ReLU(),
        #                                   Linear(dim, dim)
        #                                   )
        
        # self.n2v_encoder = Sequential(Linear(num_n2v_features, dim),
        #                               ReLU(),
        #                               Linear(dim, dim)
        #                               )
        
        self.activation_1, self.activation_l = torch.nn.PReLU(dim), torch.nn.PReLU(dim)
        self.fc_list = [Linear(dim, dim).to(device) for i in range(num_layers - 1)]
        self.activation_list = [torch.nn.PReLU(dim).to(device) for i in range(num_layers - 1)]
        
        if conv_type is None or conv_type == "GCN":
            self.conv1 = GCNConv(num_features, dim)
            self.conv_list = [GCNConv(dim, dim).to(device) for i in range(num_layers - 1)]
            
        elif conv_type == "SAGE":
            self.conv1 = SAGEConv(num_features, dim, aggr = "max")
            self.conv_list = [SAGEConv(dim, dim, aggr = "max").to(device) for i in range(num_layers - 1)]
            
        elif conv_type == "GAT":
            self.conv1 = GATConv(num_features, dim // n_attn_head, n_attn_head)
            self.conv_list = [GATConv(dim, dim // n_attn_head, n_attn_head).to(device) for i in range(num_layers - 1)]
            
        elif conv_type == "SG":
            self.conv1 = SGConv(num_features, dim, 5)
            self.conv_list = [SGConv(dim, dim, 5).to(device) for i in range(num_layers - 1)]
            
        elif conv_type == "SuperGAT":
            self.conv1 = SuperGATConv(num_features, dim // n_attn_head, n_attn_head)
            self.conv_list = [SuperGATConv(dim, dim // n_attn_head, n_attn_head).to(device) for i in range(num_layers - 1)]
            
        elif conv_type == "RGCN":
            assert "num_relations" in kwargs
            self.conv1 = RGCNConv(num_features, dim, kwargs["num_relations"], aggr = "mean")
            self.conv_list = [RGCNConv(dim, dim, kwargs["num_relations"], aggr = "max").to(device) for i in range(num_layers - 1)]
            
        else:
            raise NotImplementedError

        if self.concat:
            self.fc1 = Linear(dim * num_layers, dim)
        else:
            self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 10)

    def forward(self, x, edge_index, edge_weight, edge_type):
        xs = []
        if self.conv_type == "RGCN":
            par_edge_index, par_edge_type = dropout_adj(edge_index, edge_type, p=self.dropedge_rate, training=self.training)
            # x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.activation_1(self.conv1(x, par_edge_index, par_edge_type.view(-1)))
            xs.append(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            for conv, act, fc in zip(self.conv_list, self.activation_list, self.fc_list):
                par_edge_index, par_edge_type = dropout_adj(edge_index, edge_type, p=self.dropedge_rate, training=self.training)
                x = fc(act(conv(x, par_edge_index, par_edge_type.view(-1))))
                xs.append(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        elif self.conv_type in ["GCN", "SG"]:
            par_edge_index, par_edge_weight = dropout_adj(edge_index, edge_weight, p=self.dropedge_rate, training=self.training)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.activation_1(self.conv1(x, par_edge_index, par_edge_weight.view(-1)))
            xs.append(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            for conv, act, fc in zip(self.conv_list, self.activation_list, self.fc_list):
                par_edge_index, par_edge_weight = dropout_adj(edge_index, edge_weight, p=self.dropedge_rate, training=self.training)
                x = fc(act(conv(x, par_edge_index, par_edge_weight.view(-1))))
                xs.append(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        else:
            par_edge_index = dropout_adj(edge_index, p=self.dropedge_rate, training=self.training)[0]
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            x = self.activation_1(self.conv1(x, par_edge_index))
            xs.append(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            for conv, act, fc in zip(self.conv_list, self.activation_list, self.fc_list):
                par_edge_index = dropout_adj(edge_index, p=self.dropedge_rate, training=self.training)[0]
                x = fc(act(conv(x, par_edge_index)))
                xs.append(x)
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
                
        
        if self.concat:
            x = self.activation_l(self.fc1(torch.cat(xs, dim=-1)))
        else:
            x = self.activation_l(self.fc1(x))
        
        x = F.dropout((x), p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

model = Net(num_features=features.size(1),
            hidden_units=128,
            num_layers=3,
            dropout_rate=0.3,
            dropedge_rate=0.25,
            conv_type=args.conv,
            num_relations=num_relations,
            concat=args.concat,
            ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# loss_func = F1_Loss().to(device)
if args.weighted_loss:
    loss_func = torch.nn.NLLLoss(weight = loss_coef).cuda()
else:
    loss_func = torch.nn.NLLLoss().cuda()

def train(epoch):
    model.train()

    optimizer.zero_grad()
    output = model(data.x, data.edge_index, data.edge_weight, data.edge_type)
    # output = model(data.x, data.edge_index)
    
    loss = loss_func(output[idx_train], data.y[idx_train])
    loss.backward()
    loss_all = loss.item()
    optimizer.step()
    
    return loss_all


def evaluate(idx):
    model.eval()

    with torch.no_grad():
        class_corr, class_tot, pred_tot = np.zeros((10, ), dtype = int), np.zeros((10, ), dtype = int), np.zeros((10, ), dtype = int)
        output = model(data.x, data.edge_index, data.edge_weight, data.edge_type)
        # output = model(data.x, data.edge_index)
        pred = output.max(dim=1)[1]
        
        loss = loss_func(output[idx], data.y[idx])
        loss_all = loss.item()
        correct = pred[idx].eq(data.y[idx]).sum().item()
        total = data.y[idx].size(0)
        
        for c in range(10):
            class_corr[c] = torch.logical_and(data.y[idx] == c, pred[idx] == c).sum().item()
            class_tot[c] = (data.y[idx] == c).sum().item()
            pred_tot[c] = (pred[idx] == c).sum().item()
        
    return loss_all, correct / total, pd.DataFrame(np.vstack([class_corr, class_tot, pred_tot]),
                                                   index = ["Correct", "Total", "Pred_total"],
                                                   columns = range(10),
                                                   )


def test():
    model.eval()

    with torch.no_grad():
        output = model(data.x, data.edge_index, data.edge_weight, data.edge_type)
        # output = model(data.x, data.edge_index)
    
    # return output[torch.cat([idx_train, idx_val, idx_test])]
    return output[:24251]   # number of papers

best_acc, best_epoch = 0, 0
if args.load_model != "":
    print("Loading model weights...")
    # now_time = args.load_model
    filename = args.load_model
    model.load_state_dict(torch.load("models/saves/{}_{}.pkl".format(args.conv, args.load_model)))
    best_train_info = evaluate(idx_train)[2]
    _, best_acc, best_val_info = evaluate(idx_val)

for epoch in range(1, n_epochs + 1):
    train_loss = train(epoch)
    train_loss, train_acc, train_cls_info = evaluate(idx_train)
    val_loss, val_acc, val_cls_info = evaluate(idx_val)
    
    if val_acc > best_acc:
        best_epoch = epoch
        best_acc = val_acc
        best_train_info = train_cls_info
        best_val_info = val_cls_info
        torch.save(model.state_dict(), "models/saves/{}_{}.pkl".format(args.conv, now_time))
    
    if epoch % 20 == 0:
        print('Epoch: {:04d}, Train Loss: {:.4f}, '
              'Train Acc: {:.4f}, Val Loss: {:.4f}, '
              'Val Acc: {:.4f}'.format(epoch, train_loss, train_acc, val_loss, val_acc))

print("Optimization Finished!")
print("best epoch: {:d} with best val_acc: {:.4f}".format(best_epoch, best_acc))
print("Best training example distribution:")
print(best_train_info)
print("Best validation example distribution:")
print(best_val_info)

model.load_state_dict(torch.load("models/saves/{}_{}.pkl".format(args.conv, now_time)))
output = test()
output = output.cpu()

print('Val_(F1, precision, recall):',compute_f1_score(output[idx_val], labels[idx_val]))
print('Train_(F1, precision, recall):',compute_f1_score(output[:4844], labels))

print(output[0:5, :])

labels = labels.cpu()
result = get_paper_label(output, labels)

filename = str(now_time) +'_'+ str(round(best_acc,4))
np.save("models/probs/{}_{}.npy".format(args.conv, filename), output.numpy())
create_csv_from_result(result, submission_version=filename)
filter_csv(submission_version=filename)
