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

### References
1. Felix Wu, Tianyi Zhang, et al. [Simplifying Graph Convolutional Networks](http://arxiv.org/abs/1902.07153).  
2. Yu Rong, Wenbing Huang, et al. [DropEdge: Towards Deep Graph Convolutional Networks on Node Classification](https://arxiv.org/abs/1907.10903).  
3. Muhan Zhang, and Yixin Chen. [Link Prediction Based on Graph Neural Networks](https://arxiv.org/abs/1802.09691).  


<!-- * Node
    >1. node classification的结果可以在node目录下直接运行python example.py得到，default的GNN是"SGC"。
    >2. 如果想要使用不同的GNN Kernel，可以在运行时添加参数，例如想要使用GCN时，可用如下命令：python example.py --conv GCN。
    >3. 可以在node/src目录下的utils.py文件中的load_data函数中修改features的类型。
    >4. 可以在node/src目录下的utils.py文件中的load_edges函数中修改边的weight。
    >5. 可以在node/example.py文件中修改model的num_layers。 -->

