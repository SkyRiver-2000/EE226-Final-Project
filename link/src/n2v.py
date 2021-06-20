import numpy as np
import pandas as pd
import scipy.sparse as sp

import time

import csrgraph as cg
from nodevectors import Node2Vec

import torch
from utils import *
from seal_utils import *

class N2V(Node2Vec):
    """
    Parameters
    ----------
    p : float
        p parameter of node2vec
    q : float
        q parameter of node2vec
    d : int
        dimensionality of the embedding vectors
    w : int
        length of each truncated random walk
    """
    def __init__(self, p = 1, q = 1, d = 32, w = 10):
        super().__init__(
                    n_components = d,
                    walklen = w,
                    epochs = 50,
                    return_weight = 1.0 / p,
                    neighbor_weight = 1.0 / q,
                    threads = 1,
                    w2vparams = {'window': 5,
                                 'negative': 10, 
                                 'iter': 10,
                                 'ns_exponent': 0.5,
                                 'batch_words': 128})

N_DIM = 128
N_TOTAL_NODES = 24251 + 42614

# Load data
# adj, features, labels, idx_train, idx_val, idx_test, loss_coef = load_data()
edge_list, edge_weight, edge_type = load_edges()

# citation_list = sample_author_citation_edges()[0]
# edge_list_with_citation = np.vstack([edge_list, citation_list])
# edge_weight_with_citation = np.vstack([edge_weight, np.ones((citation_list.shape[0], 1))])

# train_edge_info = pd.DataFrame(
#     np.hstack([edge_list_with_citation, edge_weight_with_citation]),
#     columns=['src', 'dst', 'weight']
# ).drop_duplicates(subset=['src', 'dst']).values

# edge_list_with_citation = train_edge_info[:, :2]
# edge_weight_with_citation = train_edge_info[:, -1].reshape((train_edge_info.shape[0], 1))
# edge_weight_with_citation = torch.from_numpy(edge_weight_with_citation)

# citation_list = load_author_citation_edges()[0]
# graph_edge_list = np.vstack([edge_list, citation_list])
# graph_edge_list = pd.DataFrame(graph_edge_list).drop_duplicates().values
# graph_edge_list = torch.from_numpy(graph_edge_list.T).long()

# num_relations = np.shape(pd.unique(edge_type[:, 0]))[0]
# edge_list, edge_weight, edge_type, edge_list_with_citation = torch.from_numpy(edge_list.T).long(), \
#     torch.from_numpy(edge_weight).float(), torch.from_numpy(edge_type).long(), torch.from_numpy(edge_list_with_citation.T).long()
# data = Data(x = features, edge_index = edge_list, edge_label = edge_weight,
#             train_edge_index = edge_list, graph_edge_index = graph_edge_list, edge_type = edge_type, num_nodes=66865)

# data = train_val_split_edges(data)
# edge_index, _ = add_self_loops(data.graph_edge_index)
        
# data.train_neg_edge_index = negative_sampling(
#     edge_index, num_nodes=data.num_nodes,
#     num_neg_samples=data.train_pos_edge_index.size(1)
# )

# edge_list = torch.cat([data.train_pos_edge_index, data.train_neg_edge_index], dim = 1).numpy().T
print("N2V_{}d_{}t.npy".format(N_DIM, np.shape(pd.unique(edge_type[:, 0]))[0]))

G = cg.csrgraph(sp.csr_matrix((np.ones((edge_list.shape[1], )), (edge_list[0, :], edge_list[1, :])),
                              shape=(N_TOTAL_NODES, N_TOTAL_NODES), dtype=np.float32))
g2v = N2V(p=1.0, q=1.0, d=N_DIM, w=30)
embeddings = g2v.fit_transform(G)
print(embeddings.shape)

np.save("../N2V_{}d_{}t.npy".format(N_DIM, np.shape(pd.unique(edge_type[:, 0]))[0]), embeddings)
