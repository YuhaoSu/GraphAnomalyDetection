#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 22:36:32 2020

@author: suyuhao
"""
import networkx as nx

import numpy as np
import scipy.sparse as sp
"""
np.random.seed(1)
num_nodes = 30
num_clique = 2
clique_size = 5
adj = np.zeros((num_nodes,num_nodes))
num_features = 10
features = np.random.randint(100, size=(num_nodes,num_features))
feature_org = np.copy(features)

nodes_set = np.arange(num_nodes)
temp_nodes_set = set(np.copy(nodes_set))
anomaly_nodes_set = []
for i in range(num_clique):
    anomaly_nodes = np.random.choice(list(temp_nodes_set), clique_size, replace=False)
    #print(type(anomaly_nodes))
    #print("anomaly nodes",anomaly_nodes)
    for j in range(clique_size):
        for k in range(clique_size):
            adj[anomaly_nodes[j],anomaly_nodes[k]] = 1
    temp_nodes_set = temp_nodes_set - set(anomaly_nodes)
    anomaly_nodes_set.append(list(anomaly_nodes))

str_anomaly_nodes_set = [j for i in anomaly_nodes_set for j in i]
str_normal_nodes_set = list(temp_nodes_set)



    

feat_anomaly_nodes_set = np.random.choice(list(str_normal_nodes_set), clique_size*num_clique, replace=False)
remaining_normal = set(str_normal_nodes_set) - set(feat_anomaly_nodes_set)
index_set = []
for i in range(clique_size*num_clique):
    temp = np.random.choice(list(remaining_normal), k, replace=False)
    max_distance = 0
    index = 0
    for j in range(k):
        distance = np.linalg.norm(features[feat_anomaly_nodes_set[i],:] -  features[temp[j],:])
        if distance >  max_distance:
            max_distance = distance
            index = temp[j]
    index_set.append(index)
    features[feat_anomaly_nodes_set[i],:] = features[index,:]    

final_anomaly = list(set(str_anomaly_nodes_set).union(set(feat_anomaly_nodes_set)))
final_normal = list(remaining_normal)

error = np.random.randint(100,size=num_nodes)
pred_gnd = np.zeros(num_nodes)
sorted_error = np.sort(error)
sort_index = np.argsort(error)
pred_gnd[sort_index[10:]]=1
print(sort_index[10:])
pred_gnd[sort_index[num_nodes-len(final_anomaly):]]=1
print(sort_index[num_nodes-len(final_anomaly):])




a = np.load("/Users/suyuhao/Documents/AD/gae_pytorch/npzdata/amazon_electronics_computers.npz")
adj_data = a['adj_data']
adj_indices = a['adj_indices']
adj_indptr = a['adj_indptr']
adj_shape = a['adj_shape']
attr_data = a['attr_data']
attr_indices = a['attr_indices']
attr_indptr = a['attr_indptr']
attr_shape = a['attr_shape']
labels = a['labels']
class_names = a['class_names']

x = sp.csr_matrix((a['attr_data'], a['attr_indices'], a['attr_indptr']),
                      a['attr_shape']).toarray()



adj = sp.csr_matrix((a['adj_data'], a['adj_indices'], a['adj_indptr']),
                        a['adj_shape']).toarray()




def load_other_data(dataset_address):
    a = np.load(dataset_address)
    features = sp.csr_matrix((a['attr_data'], a['attr_indices'], a['attr_indptr']),
                      a['attr_shape']).toarray()
    adj = sp.csr_matrix((a['adj_data'], a['adj_indices'], a['adj_indptr']),
                        a['adj_shape']).toarray()
    #print("adj type", type(adj))
    adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
    #print("adj type", type(adj))

    return adj, features 

def anomaly_injection_adj(adj, clique_size, num_clique):
    
    if isinstance(adj,sp.csr.csr_matrix) is True:
        adj = np.array(adj.todense())
        num_nodes = adj.shape[0]
        nodes_set = np.arange(num_nodes)
        temp_nodes_set = set(np.copy(nodes_set))
        anomaly_nodes_set = []
        for i in range(num_clique):
            anomaly_nodes = np.random.choice(list(temp_nodes_set), clique_size, replace=False)
            #print("anomaly nodes",anomaly_nodes)
            for j in range(clique_size):
                for k in range(clique_size):
                    adj[anomaly_nodes[j],anomaly_nodes[k]] = 1
            temp_nodes_set = temp_nodes_set - set(anomaly_nodes)
            anomaly_nodes_set.append(list(anomaly_nodes))
        str_anomaly_nodes_set = [j for i in anomaly_nodes_set for j in i]
        str_normal_nodes_set = list(temp_nodes_set)
        adj = nx.adjacency_matrix(nx.from_numpy_array(adj))
        return str_anomaly_nodes_set, str_normal_nodes_set, adj
    else:
        raise Exception("data type not match")


def anomaly_injection_features(features, str_anomaly_nodes_set, str_normal_nodes_set, clique_size, num_clique, k):
    print(type(features))
    if isinstance(features, np.ndarray) is True:
        #features = features.numpy()
        num_nodes = features.shape[0]
        feat_anomaly_nodes_set = np.random.choice(list(str_normal_nodes_set), clique_size*num_clique, replace=False)
        remaining_normal = set(str_normal_nodes_set) - set(feat_anomaly_nodes_set)
        index_set = []
        for i in range(clique_size*num_clique):
            temp = np.random.choice(list(remaining_normal), k, replace=False)
            max_distance = 0
            index = 0
            for j in range(k):
                distance = np.linalg.norm(features[feat_anomaly_nodes_set[i],:] -  features[temp[j],:])
                if distance >  max_distance:
                    max_distance = distance
                    index = temp[j]
                index_set.append(index)
                features[feat_anomaly_nodes_set[i],:] = features[index,:]    

        final_anomaly = list(set(str_anomaly_nodes_set).union(set(feat_anomaly_nodes_set)))
        final_normal = list(remaining_normal)
        gnd = np.zeros(num_nodes)
        gnd[final_anomaly] = 1
        #features = torch.FloatTensor(features)
        #gnd = torch.Tensor(gnd)
        return final_anomaly, final_normal, features, gnd


def make_ad_dataset(dataset,clique_size, num_clique, k):
    adj, features = load_other_data(dataset)
    str_anomaly_nodes_set, str_normal_nodes_set, adj = anomaly_injection_adj(adj, clique_size, num_clique)
    final_anomaly, final_normal, features, gnd = anomaly_injection_features(features, str_anomaly_nodes_set, str_normal_nodes_set, clique_size, num_clique, k)
    return adj, features, gnd

def pred_anomaly(error,clique_size, num_clique):
    num_anomaly = clique_size* num_clique*2
    num_nodes = error.detach().numpy().shape[0]
    pred_gnd = np.zeros(num_nodes)
    sort_index = np.argsort(error.detach().numpy())
    pred_gnd[sort_index[num_nodes-num_anomaly:]]=1
    #print(sort_index[num_nodes-num_anomaly:])
    return pred_gnd

def precision(pred_gnd, gnd, clique_size, num_clique):
    #num_anomaly = clique_size* num_clique*2
    num_nodes = gnd.detach().numpy().shape[0]
    count = 0
    for i in range(num_nodes):
        if pred_gnd[i] == gnd.detach().numpy()[i]:
            count = count + 1
    accuracy = count/num_nodes
    return accuracy


#adj, features = load_other_data("/Users/suyuhao/Documents/AD/gae_pytorch/npzdata/amazon_electronics_computers.npz")
adj, features, gnd = make_ad_dataset(dataset="/Users/suyuhao/Documents/AD/gae_pytorch/npzdata/amazon_electronics_computers.npz",clique_size=100, num_clique=5, k=10)

from PIL import Image

im1 = Image.open("auccora.jpg")
im2 = Image.open("f1cora.jpg")
im3 = Image.open("losscora.jpg")
im_list = [im2,im3]

pdf1_filename = "output.pdf"

im1.save(pdf1_filename, "PDF" ,save_all=True, append_images=im_list)


import torch
import torch.nn as nn
m = nn.Softmax(dim=0)
input = torch.randn(2, 3)
print("this is input", input)
output = m(input)
print("this is output", output)
"""
import torch
import torch.nn as nn
x = torch.tensor([1, 2, 3]).view(1,3)
print(x, x.size())
xp = torch.repeat_interleave(x,3, dim=0)
print(xp, xp.size())






