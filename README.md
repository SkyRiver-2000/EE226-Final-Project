# EE226-Final-Project

This repository stores the source code and some preprocessed data for the final project of EE226 in SJTU.

### Node Classification

For task 1 (node classification), just **enter the [`node`](node/) directory** and run the following command to check our results:
```shell
mkdir submission # if this directory does not exist
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
python src/seal.py
```

Some parameters can be set by passing arguments.  
For example, if you want to eliminate our ensemble learning trick, run the following command:
```shell
python src/seal.py --n_models 1
```

At present, some parameters are not supported to be set directly by passing arguments.  
If you want to revise them, check [`utils.py`](link/src/utils.py), [`seal_utils.py`](link/src/seal_utils.py) and [`seal.py`](node/src/seal.py).



<!-- * Node
    >1. node classification的结果可以在node目录下直接运行python example.py得到，default的GNN是"SGC"。
    >2. 如果想要使用不同的GNN Kernel，可以在运行时添加参数，例如想要使用GCN时，可用如下命令：python example.py --conv GCN。
    >3. 可以在node/src目录下的utils.py文件中的load_data函数中修改features的类型。
    >4. 可以在node/src目录下的utils.py文件中的load_edges函数中修改边的weight。
    >5. 可以在node/example.py文件中修改model的num_layers。 -->

