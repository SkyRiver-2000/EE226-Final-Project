import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

file_list = ["SEAL_ENSEMBLE_10x_0604130735.csv", "SEAL_ENSEMBLE_10x_0604131833.csv",
             "SEAL_ENSEMBLE_10x_0603200402.csv", "SEAL_ENSEMBLE_10x_0604121652.csv",
             "Ensemble_0604130747.csv", "SEAL_ENSEMBLE_10x_0604122530.csv"]

file_list = ["Ensemble_0606011305.csv", "Node2Vec_0619222418.csv", "Node2Vec_0620192033.csv"
             ]

# file_list = ["SEAL_0529011708.csv"]

def main():
    plt.figure()
    means = []
    results = os.listdir("submission/")
    for f in file_list:
        labels = pd.read_csv("submission/" + f)["label"].values[1000:][:50]
        plt.plot(labels, label=f)
        # plt.scatter(range(len(labels)), labels, s=200, marker='x')
        # plt.grid(linestyle='--')
        # plt.xticks(fontsize=22, fontname='Consolas')
        # plt.yticks(fontsize=22, fontname='Consolas')
        # plt.xlabel("Sample ID", fontsize=30, fontname='Consolas', labelpad=16)
        # plt.ylabel("Predicted Probability", fontsize=30, fontname='Consolas', labelpad=24)
        # plt.title("The Distribution of Sample Predictions", fontsize=30, fontname='Consolas', pad=18)
        # plt.box(on=True)
        means.append(np.mean(labels))
    plt.legend(loc = 'best')
    plt.show()
    print(means)
    
main()