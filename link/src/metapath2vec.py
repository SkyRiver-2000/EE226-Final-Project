import argparse

import torch
from torch_geometric.nn import MetaPath2Vec

from utils import *
from seal_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=128,
                    help='The number of embedding dimensions.')
parser.add_argument('--epochs', type=int, default=5,
                    help='The number of training epochs.')
args = parser.parse_args()

N_DIM = args.hidden + 1 if args.hidden % 2 == 1 else args.hidden
N_EPOCHS = args.epochs

edge_list, edge_weight, edge_type = load_author_edges()
edge_list = torch.LongTensor(edge_list.T)
edge_weight = torch.from_numpy(edge_weight)
edge_type = torch.LongTensor(edge_type)
data = Data(edge_index=edge_list, edge_weight=edge_weight, edge_label=edge_weight, edge_type=edge_type, num_nodes=N_TOTAL_AUTHORS)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = train_val_split_edges_n2v(data)

edge_index, _ = add_self_loops(data.train_edge_index)

data.train_neg_edge_index = negative_sampling(
    edge_index, num_nodes=data.num_nodes,
    num_neg_samples=data.train_pos_edge_index.size(1)
)

train_pos_edge = data.train_pos_edge_index.numpy().T
train_neg_edge = data.train_neg_edge_index.numpy().T

data.val_neg_edge_index = negative_sampling(
    edge_index, num_nodes=data.num_nodes,
    num_neg_samples=data.val_pos_edge_index.size(1)
)

data.edge_index = edge_list
val_pos_edge = data.val_pos_edge_index.numpy().T
val_neg_edge = data.val_neg_edge_index.numpy().T

train_edge, val_edge = np.vstack([train_pos_edge, train_neg_edge]), np.vstack([val_pos_edge, val_neg_edge])
train_label = np.hstack([np.ones((data.train_pos_edge_index.size(1), )), np.zeros((data.train_neg_edge_index.size(1), ))]).astype(int)
val_label = np.hstack([np.ones((data.val_pos_edge_index.size(1), )), np.zeros((data.val_neg_edge_index.size(1), ))]).astype(int)

train_label = torch.LongTensor(train_label).to(device)
val_label = torch.LongTensor(val_label).to(device)

edge_index_dict = load_het_edges()

metapath = [
    ('author', 'coauthor', 'author'),
    ('author', 'wrote', 'paper'),
    ('paper', 'written by', 'author'),
    ('author', 'coauthor', 'author'),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MetaPath2Vec(edge_index_dict,
                     embedding_dim=N_DIM//2,
                     metapath=metapath,
                     walk_length=20,
                     context_size=8,
                     walks_per_node=20,
                     num_negative_samples=3,
                     sparse=True,
                     ).to(device)

loader = model.loader(batch_size=64, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in tqdm(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test():
    model.eval()
    z = model('author')[N_TOTAL_PAPERS:]
    X_train = z[train_edge[:, 0], :] * z[train_edge[:, 1], :]
    X_val = z[val_edge[:, 0], :] * z[val_edge[:, 1], :]
    acc = model.test(X_train, train_label,
                     X_val, val_label,
                     max_iter=100)
    
    return acc


for epoch in range(1, N_EPOCHS+1):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')
    
z1 = model('author').detach().cpu().numpy()

metapath = [
    ('author', 'wrote', 'paper'),
    ('paper', 'cited', 'paper'),
    ('paper', 'written by', 'author'),
]
model = MetaPath2Vec(edge_index_dict,
                     embedding_dim=N_DIM//2,
                     metapath=metapath,
                     walk_length=20,
                     context_size=6,
                     walks_per_node=20,
                     num_negative_samples=3,
                     sparse=True,
                     ).to(device)
loader = model.loader(batch_size=64, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

for epoch in range(1, N_EPOCHS+1):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

z2 = model('author').detach().cpu().numpy()
z = np.hstack([z1, z2])
np.save("metapath2vec_{}d.npy".format(N_DIM), z)
