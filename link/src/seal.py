import math
import time
import random
import argparse
import os.path as osp
from tqdm import tqdm
from itertools import chain

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from scipy.sparse.csgraph import shortest_path
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss

from torch_geometric.nn import GCNConv, SGConv, global_sort_pool
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader
from torch_geometric.utils import (negative_sampling, add_self_loops,
                                   k_hop_subgraph, to_scipy_sparse_matrix)

from utils import *
from models import *
from seal_utils import train_val_split_edges


parser = argparse.ArgumentParser()
parser.add_argument('--load_model', type=str, default="",
                    help='The path to load trained model from local disk.')
parser.add_argument('--epochs', type=int, default=1,
                    help='The number of training epochs.')
parser.add_argument('--n_models', type=int, default=10,
                    help='The number of ensemble model to train.')
parser.add_argument('--data_division', type=int, default=1,
                    help='Each model is only assigned approximately 1/k training data.')
parser.add_argument('--author_link_only', action='store_true', default=False,
                    help='If set to True, only coauthor link will be used for training.')
parser.add_argument('--train_with_citation', action='store_true', default=False,
                    help='If set to True, citation links will be sampled and used for training.')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
adj, features, labels, idx_train, idx_val, idx_test, loss_coef = load_data()
edge_list, edge_weight, edge_type = load_edges()

citation_list = sample_author_citation_edges()[0]
edge_list_with_citation = np.vstack([edge_list, citation_list])
edge_weight_with_citation = np.vstack([edge_weight, np.ones((citation_list.shape[0], 1))])

train_edge_info = pd.DataFrame(
    np.hstack([edge_list_with_citation, edge_weight_with_citation]),
    columns=['src', 'dst', 'weight']
).drop_duplicates(subset=['src', 'dst']).values

edge_list_with_citation = train_edge_info[:, :2]
edge_weight_with_citation = train_edge_info[:, -1].reshape((train_edge_info.shape[0], 1))
edge_weight_with_citation = torch.from_numpy(edge_weight_with_citation)

citation_list = load_author_citation_edges()[0]
graph_edge_list = np.vstack([edge_list, citation_list])
graph_edge_list = pd.DataFrame(graph_edge_list).drop_duplicates().values
graph_edge_list = torch.from_numpy(graph_edge_list.T).long()

num_relations = np.shape(pd.unique(edge_type[:, 0]))[0]
edge_list, edge_weight, edge_type, edge_list_with_citation = torch.from_numpy(edge_list.T).long(), \
    torch.from_numpy(edge_weight).float(), torch.from_numpy(edge_type).long(), torch.from_numpy(edge_list_with_citation.T).long()
data = Data(x = features, edge_index = edge_list, edge_label = edge_weight,
            train_edge_index = edge_list, graph_edge_index = graph_edge_list, edge_type = edge_type, num_nodes = 66865)

class SEALDataset(InMemoryDataset):
    def __init__(self, data, num_hops, split='train'):
        self.data = data
        self.num_hops = num_hops
        
        super(SEALDataset, self).__init__('.')
        index = ['train', 'val', 'output'].index(split)
        
        print("Loading preprocessed data...")
        self.data, self.slices = torch.load(self.processed_paths[index])
        
    @property
    def raw_dir(self):
        return osp.join(self.root, 'dataset')
    
    @property
    def processed_dir(self):
        if args.train_with_citation:
            return osp.join(self.root, 'dataset', 'processed_with_pos_cite')
        if not args.author_link_only:
            return osp.join(self.root, 'dataset', 'processed_without_neg_cite_')
        return osp.join(self.root, 'dataset', 'processed_only_author_link')
    
    @property
    def raw_file_names(self):
        return ['author_paper_all_with_year.csv',
                'labeled_papers_with_authors.csv',
                'author_pairs_to_pred.csv',
                'paper_reference.csv']

    @property
    def processed_file_names(self):
        return ['SEAL_train_data.pt', 'SEAL_val_data.pt', 'SEAL_output_data.pt']

    def process(self):
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

        data = train_val_split_edges(self.data)

        edge_index, _ = add_self_loops(data.graph_edge_index)
        
        data.train_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1)
        )
        
        data.val_neg_edge_index = negative_sampling(
            edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.val_pos_edge_index.size(1)
        )
        
        pairs_to_pred = [[], []]
        for pair in pd.read_csv("dataset/author_pairs_to_pred_with_index.csv")["author_pair"].values:
            temp = pair.split(' ')
            pairs_to_pred[0].append(int(temp[0]) + N_TOTAL_PAPERS)
            pairs_to_pred[1].append(int(temp[1]) + N_TOTAL_PAPERS)
        
        data.output_edge_index = torch.LongTensor(pairs_to_pred)

        self.__max_z__ = 0

        # Collect a list of subgraphs for training, validation and test.
        
        print("Generating training subgraphs...")
        train_pos_list = self.extract_enclosing_subgraphs(
            data.train_pos_edge_index, data.train_edge_index, data.train_pos_edge_label)
        train_neg_list = self.extract_enclosing_subgraphs(
            data.train_neg_edge_index, data.train_edge_index, 0)
        
        print("Generating evaluating subgraphs...")
        val_pos_list = self.extract_enclosing_subgraphs(
            data.val_pos_edge_index, data.train_edge_index, data.val_pos_edge_label)
        val_neg_list = self.extract_enclosing_subgraphs(
            data.val_neg_edge_index, data.train_edge_index, 0)
        
        print("Generating output subgraphs...")
        output_list = self.extract_enclosing_subgraphs(
            data.output_edge_index,
            data.train_edge_index,
            # torch.cat([data.train_pos_edge_index, data.val_pos_edge_index], dim=1),
            y = 1,
        )

        # Convert labels to one-hot features.
        for data in chain(train_pos_list, train_neg_list, val_pos_list,
                          val_neg_list, output_list):
            drnl_label = F.one_hot(data.z, self.__max_z__ + 1).to(torch.float)
            data.x = drnl_label
            # data.x = torch.cat([drnl_label, data.x], dim = 1)

        torch.save(self.collate(train_pos_list + train_neg_list),
                   self.processed_paths[0])
        torch.save(self.collate(val_pos_list + val_neg_list),
                   self.processed_paths[1])
        torch.save(self.collate(output_list),
                   self.processed_paths[2])

    def extract_enclosing_subgraphs(self, link_index, edge_index, y):
        data_list, i = [], 0
        if isinstance(y, torch.Tensor):
            y = y.view(-1)
        for src, dst in tqdm(link_index.t().tolist()):
            sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                [src, dst], self.num_hops, edge_index, relabel_nodes=True)
            src, dst = mapping.tolist()

            # Remove target link from the subgraph.
            mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
            mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
            sub_edge_index = sub_edge_index[:, mask1 & mask2]

            # Calculate node labeling.
            z = self.drnl_node_labeling(sub_edge_index, src, dst,
                                        num_nodes=sub_nodes.size(0))

            data = Data(x=self.data.x[sub_nodes], z=z,
                        edge_index=sub_edge_index, y=y[i].item() if isinstance(y, torch.Tensor) else 0)
            data_list.append(data)
            
            i += 1

        return data_list

    def drnl_node_labeling(self, edge_index, src, dst, num_nodes=None):
        # Double-radius node labeling (DRNL).
        src, dst = (dst, src) if src > dst else (src, dst)
        adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True,
                                 indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True,
                                 indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = dist // 2, dist % 2

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[src] = 1.
        z[dst] = 1.
        z[torch.isnan(z)] = 0.

        self.__max_z__ = max(int(z.max()), self.__max_z__)

        return z.to(torch.long)


train_dataset = SEALDataset(data, num_hops=1, split='train')
val_dataset = SEALDataset(data, num_hops=1, split='val')
test_dataset = SEALDataset(data, num_hops=1, split='output')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)
test_loader = DataLoader(test_dataset, batch_size=512)

print("Initializing models...")
ensemble_model_list = [ASAPDefault(train_dataset, hidden=32, num_layers=3, dropout=0.3).to(device) for _ in trange(args.n_models)]
# ensemble_model_list = [UNet(train_dataset.num_features, hidden=32, num_layers=3,
#                             pool_ratios=[0.75, 0.75, int(1)]).to(device) for _ in trange(args.n_models)]
# ensemble_model_list = [DGCNN(train_dataset, hidden_channels=32, num_layers=3, GNN=GCNConv).to(device) for _ in trange(args.n_models)]
optimizer_list = [torch.optim.Adam(params=ensemble_model_list[i].parameters(), lr=0.0001, weight_decay=0.001) for i in range(args.n_models)]
# optimizer_list = [torch.optim.SGD(params=ensemble_model_list[i].parameters(),
#                                   lr=0.01, momentum=0.9, weight_decay=0.001) for i in range(args.n_models)]

train_idx = np.random.randint(0, args.n_models, size=(len(train_loader)*args.epochs*(args.n_models // args.data_division), ), dtype=int)


@torch.no_grad()
def evaluate(loader):
    for i in range(args.n_models):
        ensemble_model_list[i].eval()

    y_pred, y_true = [], []
    for data in tqdm(loader):
        data, logits = data.to(device), 0
        for i in range(args.n_models):
            logits += ensemble_model_list[i](data.x, data.edge_index, batch=data.batch)
        y_pred.append((logits / args.n_models).view(-1).cpu())
        y_true.append((data.y > 0).long().view(-1).cpu().to(torch.float))

    return roc_auc_score(torch.cat(y_true), torch.cat(y_pred[i]))


def train(epoch):
    for i in range(args.n_models):
        ensemble_model_list[i].train()
        
    loss_list, batch_cnt = [], len(train_loader)*epoch*(args.n_models // args.data_division)
    for s in range(args.n_models // args.data_division):
        total_loss = 0
        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer_list[train_idx[batch_cnt]].zero_grad()
            logits = ensemble_model_list[train_idx[batch_cnt]](data.x, data.edge_index, batch=data.batch)
            # loss = F.mse_loss(torch.sigmoid(logits.view(-1)), data.y.float()).mean()
            loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
            loss.backward()
            optimizer_list[train_idx[batch_cnt]].step()
            total_loss += loss.item() * data.num_graphs
            batch_cnt += 1
        loss_list.append(total_loss / len(train_loader))
        val_auc = evaluate(val_loader)
        print("Epoch: {}, Stage: {}, Loss: {:.4f}, Val: {:.4f}".format(epoch+1, s+1, loss_list[-1], val_auc))

    return loss_list[-1], val_auc


@torch.no_grad()
def test():
    for i in range(args.n_models):
        ensemble_model_list[i].eval()
    all_outputs = []
    
    for data in tqdm(test_loader):
        data = data.to(device)
        logits = 0
        for i in range(args.n_models):
            logits += ensemble_model_list[i](data.x, data.edge_index, batch=data.batch)
        probs = torch.sigmoid(logits / args.n_models).view(-1)
        all_outputs.append(probs.cpu().numpy())
    
    return np.hstack(all_outputs)


now_time = time.strftime("%m%d%H%M%S", time.localtime(int(time.time())))
if args.load_model != "":
    print("Loading model weights...")
    now_time = args.load_model
    for i in range(args.n_models):
        ensemble_model_list[i].load_state_dict(torch.load("models/saves/SEAL_{}_{}.pth".format(now_time, i)))
    best_val_auc = evaluate(val_loader)
    
    print(f"Loaded model has AUC: {best_val_auc:.4f}")


best_val_auc, best_loss = 0, 1e7
for epoch in range(1, args.epochs + 1):
    loss, val_auc = train(epoch-1)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        for i in range(args.n_models):
            torch.save(ensemble_model_list[i].state_dict(), "models/saves/SEAL_{}_{}.pth".format(now_time, i))
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_auc:.4f}')


for i in range(args.n_models):
    ensemble_model_list[i].load_state_dict(torch.load("models/saves/SEAL_{}_{}.pth".format(now_time, i)))
test_outputs = test()

if args.n_models > 1:
    model_type = "SEAL_ENSEMBLE_{}x".format(args.n_models)
else:
    model_type = "SEAL"
create_submission(test_outputs, model_type, now_time)

print("{}_{}.csv".format(model_type, now_time))
