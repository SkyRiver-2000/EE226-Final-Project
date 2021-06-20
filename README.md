# EE226-Final-Project

链路预测任务目前详情见 link-prediction 分支。  

节点分类任务目前详情见 tune-sgconv 分支。


* Node
    >node classification的结果可以在node目录下直接运行python example.py得到，default的GNN是"SGC"。
    
    >如果想要使用不同的GNN Kernel，可以在运行时添加参数，例如想要使用GCN时，可用如下命令：python example.py --conv GCN。
    
    >可以在node/src目录下的utils.py文件中的load_data函数中修改features的类型。
    
    >可以在node/src目录下的utils.py文件中的load_edges函数中修改边的weight。
    
    >可以在node/example.py文件中修改model的num_layers。
* Node
