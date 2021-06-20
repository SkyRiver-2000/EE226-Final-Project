import numpy as np
import pandas as pd
import scipy.sparse as sp

import time

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import torch
from utils import *
from seal_utils import *

edge_list, edge_weight, edge_type = load_author_edges()
features = np.load("author_feature.npy")

data = Data(x = features, edge_index = torch.LongTensor(edge_list.T), edge_label = torch.LongTensor(edge_weight), num_nodes = 42614)
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

val_pos_edge = data.val_pos_edge_index.numpy().T
val_neg_edge = data.val_neg_edge_index.numpy().T

train_edge, val_edge = np.vstack([train_pos_edge, train_neg_edge]), np.vstack([val_pos_edge, val_neg_edge])
train_label = np.hstack([np.ones((data.train_pos_edge_index.size(1), )), np.zeros((data.train_neg_edge_index.size(1), ))]).astype(int)
val_label = np.hstack([np.ones((data.val_pos_edge_index.size(1), )), np.zeros((data.val_neg_edge_index.size(1), ))]).astype(int)

# X_train = np.array([np.hstack([features[edge[0], :], features[edge[1], :]]) for edge in train_edge])
# X_val = np.array([np.hstack([features[edge[0], :], features[edge[1], :]]) for edge in val_edge])

# X_train = np.hstack([features[train_edge[:, 0], :], features[train_edge[:, 1], :], features[train_edge[:, 0], :] * features[train_edge[:, 1], :]])
# X_val = np.hstack([features[val_edge[:, 0], :], features[val_edge[:, 1], :], features[val_edge[:, 0], :] * features[val_edge[:, 1], :]])

X_train = features[train_edge[:, 0], :] * features[train_edge[:, 1], :]
X_val = features[val_edge[:, 0], :] * features[val_edge[:, 1], :]
eval_set = [(X_val, val_label)]

print("Data preparation done...")

# classifier = LogisticRegression()
# classifier = LinearSVC()
# classifier.fit(X=X_train, y=train_label)
classifier = LGBMClassifier(n_jobs=8, boosting_type='gbdt', reg_lambda=0.5,
                            n_estimators=1024,
                            subsample_for_bin=150000
                            )
classifier.fit(X=X_train, y=train_label, eval_set=eval_set, eval_metric='auc', verbose=20, early_stopping_rounds=5)

if isinstance(classifier, LinearSVC):
    predict_prob = lambda x: 1 / (1 + np.exp(-classifier.decision_function(x)))
else:
    predict_prob = lambda x: classifier.predict_proba(x)[:, 1]

y_val_pred = predict_prob(X_val)
val_auc = roc_auc_score(val_label, y_val_pred)
print("Validation AUC score: {:.5f}".format(val_auc))

X_test = []
for pair in pd.read_csv("dataset/author_pairs_to_pred_with_index.csv")["author_pair"].values:
    temp = pair.split(' ')
    # X_test.append(np.hstack([features[int(temp[0]), :], features[int(temp[1]), :]]))
    X_test.append(features[int(temp[0]), :] * features[int(temp[1]), :])
    # f = np.hstack([features[int(temp[0]), :], features[int(temp[1]), :], features[int(temp[0]), :] * features[int(temp[1]), :]])
    # X_test.append(f)
X_test = np.array(X_test)

test_outputs = predict_prob(X_test)
now_time = time.strftime("%m%d%H%M%S", time.localtime(int(time.time())))
create_submission(test_outputs, "Node2Vec", now_time)