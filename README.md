# EE226 Final Project

This repository stores the source code and some preprocessed data of **Group 12** for the final project of EE226 in SJTU.  
For more details please refer to [our paper](report.pdf).

### Performance
* 0.51895 mean F1-score (2nd) on private leaderboard of node classification
* 0.79756 AUC score (2nd) on private leaderboard of link prediction

Note: Combining our `SEAL` solution with `node2vec+lightgbm` solution, we achieve 0.81069 on link prediction.

### Environment
We use a conda environment with python 3.7.0. Key packages (with necessary dependencies):  
* pytorch 1.8.1
* torch-geometric 1.7.0
* lightgbm 3.1.1

Lower version of these packages might also work fine, but we only test with the specified versions above.  
Some basic packages like numpy, pandas, scipy, and tqdm are also required, and conda/pip makes things fine.

### Node Classification

For task 1 (node classification), just **enter the [`node`](node/) directory** and run the following command to check our results:
```shell
mkdir submission # if this directory does not exist
mkdir models # if this directory does not exist
mkdir models/saves # if this directory does not exist
python example.py
```

Some parameters can be set by passing arguments.  
For example, if you want to use GCN rather than default SGC as single GNN layers, run the following command:
```shell
python example.py --conv GCN
```

At present, some parameters are not supported to be set directly by passing arguments.  
If you want to revise them, check [`utils.py`](node/src/utils.py) and [`example.py`](node/example.py).

### Link Prediction

For task 2 (link prediction), just **enter the [`link`](link/) directory** and run the following command to check our results:
```shell
mkdir submission # if this directory does not exist
mkdir models # if this directory does not exist
mkdir models/saves # if this directory does not exist
python src/seal.py
```

Some parameters can be set by passing arguments.  
For example, if you want to eliminate our ensemble learning trick, run the following command:
```shell
python src/seal.py --n_models 1
```

At present, some parameters are not supported to be set directly by passing arguments.  
If you want to revise them, check [`utils.py`](link/src/utils.py), [`seal_utils.py`](link/src/seal_utils.py) and [`seal.py`](node/src/seal.py).

We also provide `node2vec` and `metapath2vec` demo:
* [`node2vec.py`](link/src/node2vec.py): Generate node embeddings using node2vec.
* [`metapath2vec.py`](link/src/metapath2vec.py): Generate node embeddings using metapath2vec.
* [`n2v_pred.py`](link/src/n2v_pred.py): Solve link prediction using node embeddings.

Trained embeddings (.npy files) are provided under the [`link`](link/) folder.  
Demo of generative models are to be updated in the future.

### References
[1] A. Grover, and J. Leskovec. [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653), KDD 2016.
[2] Y. Dong, N. Chawla, et al. [metapath2vec: Scalable Representation Learning for Heterogeneous Networks](https://dl.acm.org/doi/10.1145/3097983.3098036), KDD 2017.
[3] F. Wu, T. Zhang, et al. [Simplifying Graph Convolutional Networks](http://arxiv.org/abs/1902.07153), ICML 2019.  
[4] Y. Rong, W. Huang, et al. [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://arxiv.org/abs/1907.10903), ICLR 2020.  
[5] M. Zhang, and Y. Chen. [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691), NeurIPS 2018.  


<!-- * Node
    >1. node classification的结果可以在node目录下直接运行python example.py得到，default的GNN是"SGC"。
    >2. 如果想要使用不同的GNN Kernel，可以在运行时添加参数，例如想要使用GCN时，可用如下命令：python example.py --conv GCN。
    >3. 可以在node/src目录下的utils.py文件中的load_data函数中修改features的类型。
    >4. 可以在node/src目录下的utils.py文件中的load_edges函数中修改边的weight。
    >5. 可以在node/example.py文件中修改model的num_layers。 -->

