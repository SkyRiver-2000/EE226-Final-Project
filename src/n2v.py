import numpy as np
import pandas as pd
import scipy.sparse as sp

import time

import csrgraph as cg
from nodevectors import Node2Vec, ProNE, GGVec, Glove

# import networkx as nx
# from fastnode2vec import Graph, Node2Vec

import torch
from src.utils import load_edges, load_reference_edges

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
                    threads = 4,
                    w2vparams = {'window': 4,
                                'negative': 5, 
                                'iter': 10,
                                'ns_exponent': 0.5,
                                'batch_words': 128})

N_DIM = 256
N_TOTAL_NODES = 24251 + 42614

# Load data
edge_list, edge_weight, edge_type = load_edges()
G = cg.csrgraph(sp.csr_matrix((np.ones((edge_list.shape[0], )), (edge_list[:, 0], edge_list[:, 1])),
                              shape=(N_TOTAL_NODES, N_TOTAL_NODES), dtype=np.float32))
# G = cg.read_edgelist("dataset/edgelist.csv", directed=False, sep=',', header=0, dtype=int)
# g2v = Glove(N_DIM)
g2v = N2V(p=1.0, q=1.0, d=N_DIM, w=20)
embeddings = g2v.fit_transform(G)
print(embeddings.shape)
print("N2V_{}d_{}t".format(N_DIM, np.shape(pd.unique(edge_type[:, 0]))[0]))
np.save("../N2V_{}d_{}t.npy".format(N_DIM, np.shape(pd.unique(edge_type[:, 0]))[0]), embeddings)

# G = Graph([(x[0], x[1], w) for x, w in zip(edge_list, np.squeeze(edge_weight))], directed=False, weighted=True)

# n2v = Node2Vec(G, dim=N_DIM, walk_length=20, context=10, p=2.0, q=2.0, workers=1, seed=42)
# n2v.train(epochs=50)

# print(n2v.wv[0])
# now_time = time.strftime("%m%d%H%M%S", time.localtime(int(time.time())))
# n2v.save("../ggvec_embedding_{}d_{}t.txt.gz.wv".format(N_DIM, np.shape(pd.unique(edge_type[:, 0]))[0]))