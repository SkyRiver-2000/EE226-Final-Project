import torch
from torch_geometric.nn import Node2Vec

from utils import *
from seal_utils import *

import argparse
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=128,
                    help='The number of embedding dimensions.')
parser.add_argument('--epochs', type=int, default=5,
                    help='The number of training epochs.')
parser.add_argument('--with_paper', action='store_true', default=False,
                    help='If set to True, paper nodes will also be included in graph.')
parser.add_argument('--with_neg_edge', action='store_true', default=False,
                    help='If set to True, negative edges will also be included. (For SEAL)')
args = parser.parse_args()

N_DIM = args.hidden
N_EPOCHS = args.epochs

num_nodes = N_TOTAL_NODES if args.with_paper else N_TOTAL_AUTHORS

edge_list, edge_weight, edge_type = load_edges() if args.with_paper else load_author_edges()
edge_list = torch.LongTensor(edge_list.T)
edge_weight = torch.from_numpy(edge_weight)
edge_type = torch.LongTensor(edge_type)
data = Data(edge_index=edge_list, edge_weight=edge_weight, edge_label=edge_weight, edge_type=edge_type, num_nodes=num_nodes)

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

data.edge_index = torch.cat([data.train_pos_edge_index, data.train_neg_edge_index], dim=1) if args.with_neg_edge else edge_list

model = Node2Vec(data.edge_index,
                 embedding_dim=N_DIM,
                 walk_length=100,
                 context_size=5,
                 walks_per_node=20,
                 p=2,
                 q=2,
                 num_negative_samples=1,
                 num_nodes=data.num_nodes,
                 sparse=True,
                 ).to(device)
loader = model.loader(batch_size=64, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

val_pos_edge = data.val_pos_edge_index.numpy().T
val_neg_edge = data.val_neg_edge_index.numpy().T

train_edge, val_edge = np.vstack([train_pos_edge, train_neg_edge]), np.vstack([val_pos_edge, val_neg_edge])
train_label = np.hstack([np.ones((data.train_pos_edge_index.size(1), )), np.zeros((data.train_neg_edge_index.size(1), ))]).astype(int)
val_label = np.hstack([np.ones((data.val_pos_edge_index.size(1), )), np.zeros((data.val_neg_edge_index.size(1), ))]).astype(int)

train_label = torch.LongTensor(train_label).to(device)
val_label = torch.LongTensor(val_label).to(device)

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
    z = model()
    X_train = z[train_edge[:, 0], :] * z[train_edge[:, 1], :]
    X_val = z[val_edge[:, 0], :] * z[val_edge[:, 1], :]
    acc = model.test(X_train, train_label,
                     X_val, val_label,
                     max_iter=100)
    
    return acc

for epoch in range(1, N_EPOCHS+1):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
    
z = model().detach().cpu().numpy()
additional_data = ""

if args.with_paper:
    additional_data += "_with_paper"
if args.with_neg_edge:
    additional_data += "_with_neg_edge"
filename = "N2V_{}d_{}t{}.npy".format(N_DIM, np.shape(pd.unique(edge_type[:, 0].numpy()))[0], additional_data)
print(filename)
np.save("../{}".format(filename), z)