import time
import numpy as np
import pandas as pd

from utils import create_submission

file_list = ["Ensemble_0606011305.csv", "Node2Vec_0620193120.csv"]
file_list = ["Ensemble_0620213404.csv", "Ensemble_0620194031.csv"]
weights = [0.5, 0.5]
ensemble_labels = []

for f, w in zip(file_list, weights):
    labels = pd.read_csv("submission/" + f).values[:, 1] * w
    ensemble_labels.append(labels)

now_time = time.strftime("%m%d%H%M%S", time.localtime(int(time.time())))
create_submission(np.sum(ensemble_labels, axis = 0), "Ensemble", now_time)