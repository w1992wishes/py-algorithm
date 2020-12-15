# 【Cluster】标签传播算法

[TOC]

## 一、介绍

LPA是一种基于标签传播的局部社区划分。对于网络中的每一个节点，在初始阶段，Label Propagation算法对于每一个节点都会初始化一个唯一的一个标签。每一次迭代都会根据与自己相连的节点所属的标签改变自己的标签，更改的原则是选择与其相连的节点中所属标签最多的社区标签为自己的社区标签，这就是标签传播的含义了。随着社区标签不断传播。最终，连接紧密的节点将有共同的标签。

LPA认为每个结点的标签应该和其大多数邻居的标签相同，将一个节点的邻居节点的**标签中数量最多（或者说想加后权重最高）**的标签作为该节点自身的标签（bagging思想）。给每个节点添加标签（label）以代表它所属的社区，并通过标签的“传播”形成同一个“社区”内部拥有同一个“标签”。

**在基本思想上，LPA 和 Kmean 本质非常类似，在 LPA 的每轮迭代中，节点被归属于哪个社区，取决于其邻居中累加权重最大的label（取数量最多的节点列表对应的label是weight=1时的一种特例），而 Kmeans的则是计算和当前节点“最近”的社区，将该节点归入哪个社区。**

**但是这两个算法还是有细微的区别的:**

```
1. 首先: Kmeans是基于欧式空间计算节点向量间的距离的，而LPA则是根据节点间的“共有关系”以及“共有关系的强弱程度”来度量度量节点间的距离；
2. 第二点: Kmeasn中节点处在欧式空间中，它假设所有节点之间都存在“一定的关系”，不同的距离体现了关系的强弱。但是 LPA 中节点间只有满足“某种共有关系”时，才存在节点间的边，没有共有关系的节点是完全隔断的，计算邻居节点的时候也不会计算整个图结构，而是仅仅计算和该节点有边连接的节点，从这个角度看，LPA 的这个图结构具有更强的社区型；
```

## 二、优缺点

优点：

```
1. LPA算法的最大的优点就是算法的逻辑非常简单，相对于优化模块度算法的过程是非常快的，不用pylouvain那样的多次迭代优化过程。
2. LPA算法利用自身的网络的结构指导标签传播，这个过程是无需任何的任何的优化函数，而且算法初始化之前是不需要知道社区的个数的，随着算法迭代最后可以自己知道最终有多少个社区。
```

缺点：

划分结果不稳定，随机性强是这个算法致命的缺点。具体体现在：

```
1. 更新顺序：节点标签更新顺序随机，但是很明显，越重要的节点越早更新会加速收敛过程；
2. 随机选择：如果一个节点的出现次数最大的邻居标签不止一个时，随机选择一个标签作为自己标签。这种随机性可能会带来一个雪崩效应，即刚开始一个小小的聚类错误会不断被放大。不过话也说话来，如果相似邻居节点出现多个，可能是weight计算的逻辑有问题，需要回过头去优化weight抽象和计算逻辑；
```

## 三、算法过程

### 3.1、算法过程描述

**第一步：**先给每个节点分配对应标签，即节点1对应标签1，节点i对应标签i； 
**第二步：**遍历N个节点（for i=1：N），找到对应节点邻居，获取此节点邻居标签，找到出现次数最大标签，若出现次数最多标签不止一个，则随机选择一个标签替换成此节点标签；
**第三步：**若本轮标签重标记后，节点标签不再变化（或者达到设定的最大迭代次数），则迭代停止，否则重复第二步  

### 3.2、边权重计算

社区图结构中边的权重代表了这两个节点之间的的“关系强弱”，这个关系的定义取决于具体的场景，例如：

```
1. 两个DNS域名共享的client ip数量；
2. 两个微博ID的共同好友数量；
```

### 3.3、标签传播方式

LPA标签传播分为两种传播方式，同步更新，异步更新。

#### 1. 同步更新

同步的意思是实时，即时的意思，每个节点label更新后立即生效，其他节点在统计最近邻社区的时候，永远取的是当前图结构中的最新值。

对于节点 x，在第 t 轮迭代时，根据其所在节点在第t-1代的标签进行更新。

需要注意的是，这种同步更新的方法会存在一个问题，当遇到二分图的时候，会出现标签震荡，如下图：

![img](..\images\algorithm\lpa-case.png)

这种情况和深度学习中SGD在优化到全局最优点附近时会围绕最优点附近进行布朗运动（震荡）的原理类似。解决的方法就是设置最大迭代次数，提前停止迭代。

#### 2. 异步更新

异步更新方式可以理解为取了一个当前社区的快照信息，基于上一轮迭代的快照信息来进行本轮的标签更新。

## 四、代码实现

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string


def loadData(filePath):
    f = open(filePath)
    node_dict = {}
    edge_list_dict = {}
    for line in f.readlines():
        lines = line.strip().split("\t")
        for i in range(2):
            if lines[i] not in node_dict:
                node_dict[lines[i]] = (lines[i])  # 每个节点的社区标签就是自己
                edge_list = []  # 存放每个节点的所有邻居节点及边权重
                if len(lines) == 3:
                    edge_list.append(lines[1 - i] + ":" + lines[2])
                else:
                    edge_list.append(lines[1 - i] + ":" + "1")  # 无权图默认权重是1
                edge_list_dict[lines[i]] = edge_list
            else:
                edge_list = edge_list_dict[lines[i]]
                if len(lines) == 3:
                    edge_list.append(lines[1 - i] + ":" + lines[2])
                else:
                    edge_list.append(lines[1 - i] + ":" + "1")
                edge_list_dict[lines[i]] = edge_list
    return node_dict, edge_list_dict


def get_max_community_label(node_dict, adjacency_node_list):
    label_dict = {}
    for node in adjacency_node_list:
        node_id_weight = node.strip().split(":")
        node_id = node_id_weight[0]
        node_weight = int(node_id_weight[1])

        # 按照label为group维度，统计每个label的weight累加和
        if node_dict[node_id] not in label_dict:
            label_dict[node_dict[node_id]] = node_weight
        else:
            label_dict[node_dict[node_id]] += node_weight

    sort_list = sorted(label_dict.items(), key=lambda d: d[1], reverse=True)  # 返回weight累加和最大的社区标签
    return sort_list[0][0]


def check(node_dict, edge_dict):
    for node in node_dict.keys():
        adjacency_node_list = edge_dict[node]  # 获取该节点的邻居节点
        node_label = node_dict[node]  # 获取该节点当前label
        label = get_max_community_label(node_dict, adjacency_node_list)  # 从邻居节点列表中选择weight累加和最大的label
        if node_label >= label:
            continue
        else:
            return 0  # 找到weight权重累加和更大的label
    return 1


def label_propagation(node_dict, edge_list_dict, iteration_max):
    t = 0
    while True:
        if t > iteration_max:
            break
        # 收敛判定阶段
        if (check(node_dict, edge_list_dict) == 0):
            t = t + 1
            print('iteration: ', t)
            # 每轮迭代都更新一遍所有节点的社区label
            for node in node_dict.keys():
                # adjacency_node_list存放该节点的所有邻居节点及边权重
                adjacency_node_list = edge_list_dict[node]
                # 传播阶段，将所有邻居节点最大的社区标签更新为自己的社区标签
                node_dict[node] = get_max_community_label(node_dict, adjacency_node_list)
        else:
            break
    return node_dict


'''
第一步：先给每个节点分配对应标签，即节点1对应标签1，节点i对应标签i； 
第二步：遍历N个节点（for i=1：N），找到对应节点邻居，获取此节点邻居标签，找到出现次数最大标签，若出现次数最多标签不止一个，则随机选择一个标签替换成此节点标签；
第三步：若本轮标签重标记后，节点标签不再变化（或者达到设定的最大迭代次数），则迭代停止，否则重复第二步   
'''
if __name__ == '__main__':
    # 加载文件
    filePath = '../data/lpg.txt'

    print("初始化阶段开始！")
    # node_dict表示每个节点归属社区的状态，key是node，value是社区标签。
    # edge_list表示每个节点的所有所有邻居节点及边权重，key是node,value是一个list,list中的每个item就是该node的一个邻居节点及边权重。
    node_dict, edge_list_dict = loadData(filePath)
    iteration_max = 1000  # 设置最大迭代次数
    print("初始化阶段结束！")

    print("标签传播算法开始！")
    node_dict = label_propagation(node_dict, edge_list_dict, iteration_max)
    print("标签传播算法结束！")
    print("最终的结果为：")
    print(node_dict)
```

