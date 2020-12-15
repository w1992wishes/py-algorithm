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
