import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

import csv
import pandas as pd
from tqdm import tqdm, trange
    
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

N_TOTAL_PAPERS = 24251
N_TOTAL_AUTHORS = 42614
N_TOTAL_NODES = N_TOTAL_PAPERS + N_TOTAL_AUTHORS


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_reference_edges(path="dataset/"):
    print('Loading edge list...')
    
    reference_links = np.load("../edge_and_weight_01.npy")
    # reference_links = np.vstack([reference_links, np.fliplr(reference_links)])
    # reference_links = pd.DataFrame(reference_links).drop_duplicates().values
    reference_edge_weight = np.expand_dims(reference_links[:, -1], 1)
    reference_edge_type = np.zeros((reference_links.shape[0], 1), dtype = int)
    
    # pd.DataFrame(reference_links, columns=['src', 'dst', 'weight']).to_csv(path + "reference_edgelist.csv", index=False)
    reference_links = reference_links[:, :-1]
    
    return reference_links, reference_edge_weight, reference_edge_type


def count_citation(path="dataset/"):
    print("Running citation counting...")
    referenced = pd.read_csv(path + "paper_reference.csv").values[:, -1]
    return pd.Series(referenced).value_counts()


def load_edges(path="dataset/"):
    print('Loading edge list...')
    
    reference_links = pd.read_csv(path + "paper_reference.csv").values
    reference_links = np.vstack([reference_links, np.fliplr(reference_links)])
    reference_links = pd.DataFrame(reference_links).drop_duplicates().values
    reference_edge_weight = np.ones((reference_links.shape[0], 1), dtype = float)
    reference_edge_type = np.zeros((reference_links.shape[0], 1), dtype = int)
    
    author_paper_links = pd.read_csv(path + "author_paper_all_with_year.csv").values[:, 0:-1]
    author_paper_links[:, 0] += N_TOTAL_PAPERS
    author_paper_links = np.vstack([author_paper_links, np.fliplr(author_paper_links)])
    # author_paper_edges = np.hstack([author_paper_links, np.ones((author_paper_links.shape[0], 1))])
    author_paper_edges = np.hstack([author_paper_links, np.load(path + "author_paper_edge_weight.npy")])
    author_paper_edges = pd.DataFrame(author_paper_edges, columns=['i', 'j', 'w']).drop_duplicates(subset=['i', 'j']).values
    author_paper_links = author_paper_edges[:, 0:-1]
    # author_paper_edge_weight = np.ones((author_paper_links.shape[0], 1))
    author_paper_edge_weight = np.expand_dims(author_paper_edges[:, -1], 1) / author_paper_edges[:, -1].mean()
    author_paper_edge_type = np.ones((author_paper_links.shape[0], 1), dtype = int)
    
    coauthor_links = np.load(path + "coauthor.npy").astype(int) + N_TOTAL_PAPERS
    coauthor_links = np.vstack([coauthor_links, np.fliplr(coauthor_links)])
    coauthor_edges = pd.DataFrame(coauthor_links).value_counts()
    coauthor_links = np.asarray(list(coauthor_edges.index))
    # coauthor_edge_weight = np.ones((coauthor_links.shape[0], 1))
    coauthor_edge_weight = np.expand_dims(np.asarray(list(coauthor_edges.values)), 1) / coauthor_edges.values.mean()
    coauthor_edge_type = 2 * np.ones((coauthor_links.shape[0], 1), dtype = int)
    
    # same_author_links = np.load(path + "paper_same_author.npy")
    # same_author_links = np.vstack([same_author_links, np.fliplr(same_author_links)])
    # same_author_links = pd.DataFrame(same_author_links).drop_duplicates().values
    # same_author_edge_type = 3 * np.ones((same_author_links.shape[0], 1), dtype = int)
    
    edges_unordered = np.vstack([reference_links, author_paper_links, coauthor_links])
    edges_weight = np.vstack([reference_edge_weight, author_paper_edge_weight, coauthor_edge_weight])
    # pd.DataFrame(np.hstack([edges_unordered, edges_weight]), columns=['src', 'dst', 'weight']).to_csv(path + "edgelist.csv", index=False)
    edges_type = np.vstack([reference_edge_type, author_paper_edge_type, coauthor_edge_type])

    return edges_unordered, edges_weight, edges_type


def load_data(path="dataset/"):
    """Load citation network dataset (cora only for now)"""
    # print('Loading dataset...')

    # build graph
    edges, edge_weight, edge_type = load_edges()
    # print(edges.shape, edge_weight.shape, edge_type.shape)
    adj = sp.coo_matrix((edge_weight[:, 0], (edges[:, 0], edges[:, 1])),
                        shape=(N_TOTAL_NODES, N_TOTAL_NODES),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # print(np.shape(pd.unique(edge_type[:, 0]))[0])
    
    paper_label = pd.read_csv(path + "labeled_papers_with_authors.csv")["label"].values
    labels = encode_onehot(paper_label)
    
    idx_train, idx_val, _, _ = train_test_split(np.arange(len(paper_label)), labels, test_size=0.05, random_state=1)
    idx_test = range(len(paper_label), N_TOTAL_PAPERS)
    
    # features = np.zeros((N_TOTAL_PAPERS, 10))
    # features[:len(paper_label), :] = labels
    # features[idx_train, :] = labels[idx_train, :]
    
    publication_year = pd.read_csv(path + "author_paper_all_with_year.csv").drop_duplicates(subset=["paper_id"]).values[:, -1]
    
    # extra_features = pd.read_csv(path + "node_extra_features.csv").values
    # features = np.hstack([extra_features, encode_onehot(publication_year)])
    
    features = np.load("../N2V_512d_2t.npy")
    features = torch.FloatTensor(features)

    # features = np.load("../N2V_512d_{}t.npy".format(np.shape(pd.unique(edge_type[:, 0]))[0]))
    # features = KeyedVectors.load("../n2v_embedding_512d_3t.txt.gz.wv")
    # features = KeyedVectors.load("../n2v_embedding_512d_{}t.txt.gz.wv".format(np.shape(pd.unique(edge_type[:, 0]))[0]), mmap='r')
    # features = np.array([features.wv[i] for i in trange(N_TOTAL_NODES)])
    # features = torch.load("../dgi_embed_96d.pkl").squeeze().cpu().numpy()
    # features = sp.csr_matrix(features, dtype=np.float32)
    
    # print(features)
    
    class_tot = np.sum(labels, axis = 0)
    loss_coef = torch.from_numpy(np.mean(class_tot) / class_tot).float()

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(features)
    # features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, loss_coef


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_paper_label(test_output, given_labels):
    result = np.zeros((24251, ))
    result[:len(given_labels)] = given_labels

    preds = test_output.max(1)[1]
    result[len(given_labels):] = preds[len(given_labels):]
    
    return result

def create_csv_from_result(result, submission_version='0'):
    
    author_paper_dic = np.load('author_paper_dic.npy',allow_pickle=True).item()     # dictionary like {94: [25, 21083]}

    # transform to an "author with label" version
    author_label_dic = {}                                                           # dictionary like {0: [0, 1, 5]}
    for key in author_paper_dic:
        for index in author_paper_dic[key]:
            if key not in author_label_dic:
                author_label_dic[key] = [int(result[index])]
            else:
                if int(result[index]) not in author_label_dic[key]:
                    author_label_dic[key].append(int(result[index]))
    
    unfiltered_submission_name = 'submission/unfiltered_submission_'+submission_version+'.csv'
    f = open(unfiltered_submission_name,'w',encoding='utf-8',newline='' "")

    csv_writer = csv.writer(f)
    csv_writer.writerow(["author_id","labels"])
    for key in author_label_dic:
        csv_writer.writerow([key,' '.join([str(x) for x in author_label_dic[key]])])

    f.close()

def filter_csv(submission_version='0'):
    
    test_set = pd.read_csv("dataset/authors_to_pred.csv")
    test_authors = test_set.values.reshape((37066, ))

    unfiltered_submission_name = 'submission/unfiltered_submission_'+submission_version+'.csv'
    submission_name = 'submission/submission_'+submission_version+'.csv'
    submit = pd.read_csv(unfiltered_submission_name)
    submit.loc[test_authors].to_csv(submission_name, index = False)
    print('--------------')

def compute_f1_score(outputs, labels):

    preds = F.one_hot(outputs.max(1)[1]).cpu().numpy()[:4844]
    preds = preds.reshape((preds.shape[0] * 10,))
    labels = F.one_hot(labels).cpu().numpy()
    labels = labels.reshape((labels.shape[0] * 10,))
    print(precision_score(labels, preds), recall_score(labels, preds))
    return f1_score(labels, preds)