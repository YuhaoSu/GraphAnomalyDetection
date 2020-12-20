#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:05:59 2020

@author: suyuhao
"""

#from __future__ import division
#from __future__ import print_function

import argparse
import time
import torch
import scat
import metis

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import optim
from backup.model import GCNModelScatVAE
from backup.utils import  preprocess_graph, make_ad_dataset, pred_anomaly, precision
from sklearn.metrics import roc_auc_score, f1_score 
from backup.optimizer import scat_ae_loss_function
from fms import FMS

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=800, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=float, default=0.5, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=float, default=0.3, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.004, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--clique_size', type=int, default=20, help='create anomaly')
parser.add_argument('--num_clique', type=int, default=15, help='create anomaly')
parser.add_argument('--k', type=int, default=10, help='compare for feature anomaly')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--dim_reduce', type=int, default=0, help='0 for pca, 1 for fms, 2 for no dim reduction')    
parser.add_argument('--num_clusters', type=int, default=2, help='number of clusters')    
parser.add_argument('--alpha', type=float, default=1, help='weight parameter for feature error')  
parser.add_argument('--beta', type=float, default=1, help='weight parameter for structure error')  
parser.add_argument('--cuda', action='store_true', help='enables cuda')


args = parser.parse_args()

def dim_reduction(A, pca=True, num_of_components=128):
    if not pca:
        num_of_components = A.shape[1]
    pca = PCA(n_components=num_of_components)
    A_pca = pca.fit_transform(A)
    scaler = StandardScaler()
    for i in range(np.shape(A_pca)[0]):
        A_pca[i,:] = scaler.fit_transform(A_pca[i,:].reshape(-1,1)).reshape(-1)
    return A_pca

args.device = None
if not args.cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
   
def subgraph_anomaly_predict(error, _sg_gnd):
    sg_pred_gnd = np.zeros(len(_sg_gnd))
    sort_index = np.argsort(error)
    num_anomalies = np.count_nonzero(_sg_gnd == 1)
    sg_pred_gnd[sort_index[len(_sg_gnd)-num_anomalies:]]=1
    #print(sort_index[num_nodes-num_anomaly:])
    return sg_pred_gnd

def combine_sg_pred_gnd(clusters, num_nodes, sg_pred_, sg_nodes):
    pred_gnd = np.zeros(num_nodes)
    anomaly_index = []
    for cluster in range(clusters):
        sg_anomaly_index = []
        for node in range(len(sg_pred_[cluster])):
            if sg_pred_[cluster][node] == 1:
                sg_anomaly_index.append(node)
        anomaly_index = anomaly_index + list(np.array(sg_nodes[cluster])[sg_anomaly_index])
    print(len(anomaly_index))
    pred_gnd[anomaly_index] = 1
    
    return pred_gnd

def gae_ad(args):
    #initialize
    arg_dim = ['PCA', 'FMS','No reduction']
    print()
    print("Initializing")
    print(args)
    seed = np.random.randint(1000)
    torch.manual_seed(seed)
    print("random seed:",seed)
    loss_plot = []
    auc_plot = []
    f1_plot = []
    accuracy_plot = []
    feature_reconstruction_loss_plot = []
    structure_reconstruction_loss_plot = []
    
    #obtain basic info
    adj, features, gnd = make_ad_dataset(args.dataset, args.clique_size, args.num_clique, args.k)
    origin_features = torch.FloatTensor(features).to(args.device)
    gnd = torch.Tensor(gnd).to(args.device)
    feat_dim = features.shape[1]
    hidden1 = int(feat_dim * args.hidden1)
    hidden2 = int(feat_dim * args.hidden2)
    
    # Data preprocessing
    indices, values, shape = preprocess_graph(adj)
    adj_norm = torch.sparse.FloatTensor(indices, values, shape).to(args.device)
    print("feature_dim:",feat_dim, "hidden1_dim:",hidden1, "hidden2_dim:",hidden2,"nodes_num:",adj_norm.shape[0], "anomaly_num:", 2*args.clique_size*args.num_clique)
    print()      
    print("Start modeling")
    
    #graph clustering
    sg_nodes = {}
    sg_edges = {}
    sg_features = {}
    sg_gnd = {}

    graph = nx.from_scipy_sparse_matrix(adj)
    (st, parts) = metis.part_graph(graph, args.num_clusters)
    clusters = list(set(parts))
    cluster_membership = {node: membership for node, membership in enumerate(parts)}
    for cluster in clusters:
        subgraph = graph.subgraph([node for node in sorted(graph.nodes()) if cluster_membership[node] == cluster])
        sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
        mapper = {node: i for i, node in enumerate(sorted(sg_nodes[cluster]))}
        sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
        sg_features[cluster] = features[sg_nodes[cluster],:]
        sg_gnd[cluster] = (gnd[sg_nodes[cluster]])
    
    scat_list = []
    for cluster in clusters:
        print("cluster:", cluster)
        
    #Start wavelet transform
        print("using wavelet scattering transform")
        A = nx.from_edgelist(sg_edges[cluster])
        L = nx.linalg.laplacianmatrix.laplacian_matrix(A)
        lamb, V = np.linalg.eigh(L.toarray())
        y_features_scat = scat.getRep(sg_features[cluster], lamb, V, layer=3)
        print("y_features_scat shape after scatting", y_features_scat.shape)
        scat_list.append(y_features_scat)
    y_features_scat = np.concatenate(scat_list)
    print()
    print("y_features shape after combining", y_features_scat.shape)

# Start dim reduction
    if arg_dim[args.dim_reduce] == 'PCA':
        print("using PCA to reduce dim")
        y_features_reduced = dim_reduction(y_features_scat, num_of_components=hidden2)
    if arg_dim[args.dim_reduce] == 'FMS':
        print("using FMS to reduce dim")
        fms_ = FMS(n_components=hidden2)
        y_features_reduced = fms_.fit_transform(y_features_scat)
    if arg_dim[args.dim_reduce] == 'No reduction':
        hidden2 = y_features_scat.shape[1]
        y_features_reduced = y_features_scat   
         
    print("y_features shape after "+'{}'.format(arg_dim[args.dim_reduce]), y_features_reduced.shape)
    #Start training
    print()
    y_features = torch.FloatTensor(y_features_reduced).to(args.device)
    model = GCNModelScatVAE(feat_dim, hidden1, hidden2, args.dropout).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if torch.cuda.is_available() is True: 
        #If using cuda, then need to transfer data from GPU to CPU to save the results.
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            feature_decoder_layer_2, structure_decoder_layer_2 = model(y_features, adj_norm)
            error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss = scat_ae_loss_function(feature_decoder_layer_2,
                                                                                                                  structure_decoder_layer_2, 
                                                                                                                  origin_features, adj_norm, args.alpha,
                                                                                                                  args.beta)
            total_loss.backward()
            cur_loss = total_loss.item()
            feature_reconstruction_loss = feature_reconstruction_loss.item()
            structure_reconstruction_loss = structure_reconstruction_loss.item()
            optimizer.step()
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "time=", "{:.5f}".format(time.time() - t))
                
            loss_plot.append(cur_loss)
            feature_reconstruction_loss_plot.append(feature_reconstruction_loss)
            structure_reconstruction_loss_plot.append(structure_reconstruction_loss)

            auc = roc_auc_score(gnd.cpu().detach().numpy(), error.cpu().detach().numpy())
            pred_gnd = pred_anomaly(error.cpu().detach().numpy(), args.clique_size, args.num_clique)
            accuracy = precision(pred_gnd, gnd.cpu().detach().numpy())
            f1 = f1_score(gnd.cpu().detach().numpy(), pred_gnd)
            accuracy_plot.append(accuracy)
            auc_plot.append(auc)
            f1_plot.append(f1)
    else: 
        #CPU case
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            feature_decoder_layer_2, structure_decoder_layer_2 = model(y_features, adj_norm)
            error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss = scat_ae_loss_function(feature_decoder_layer_2,
                                                                                                                  structure_decoder_layer_2, 
                                                                                                                  origin_features, adj_norm, args.alpha,
                                                                                                                  args.beta)
            total_loss.backward()
            cur_loss = total_loss.item()
            feature_reconstruction_loss = feature_reconstruction_loss.item()
            structure_reconstruction_loss = structure_reconstruction_loss.item()
            optimizer.step()
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "time=", "{:.5f}".format(time.time() - t))
                
            loss_plot.append(cur_loss)
            feature_reconstruction_loss_plot.append(feature_reconstruction_loss)
            structure_reconstruction_loss_plot.append(structure_reconstruction_loss)
            
            auc = roc_auc_score(gnd.detach().numpy(), error.detach().numpy())
            pred_gnd = pred_anomaly(error.detach().numpy(), args.clique_size, args.num_clique)
            accuracy = precision(pred_gnd, gnd.detach().numpy())
            f1 = f1_score(gnd.detach().numpy(), pred_gnd)
            
            accuracy_plot.append(accuracy)
            auc_plot.append(auc)
            f1_plot.append(f1)
    
    # save the results
    result = {'total_loss': loss_plot, 
              'feature_reconstruction_loss': feature_reconstruction_loss,
              'structure_reconstruction_loss': structure_reconstruction_loss,
              'accuracy': accuracy_plot,
              'auc score': auc_plot,
              'f1 score': f1_plot}
    result_df = pd.DataFrame(data=result)
    result_df.shape
    result_df.csv_path = 'scat_dataset_'+'{}'.format(args.dataset)+'_hidden2_'+'{}'.format(args.hidden2)+'_anomaly_{}'.format(2*args.clique_size*args.num_clique)+'_{}'.format(arg_dim[args.dim_reduce])+'.csv'
    result_df.to_csv(result_df.csv_path)
    #os.chdir('output')
    '''
    if not os.path.exists("/Users/suyuhao/Documents/AD/gae_pytorch/output/{}".format(args.dataset)):
        os.makedirs('{}'.format(args.dataset))
    shutil.move('scat_dataset_'+'{}'.format(args.dataset)+'_hidden2_'+'{}'.format(args.hidden2)+'_anomaly_{}'.format(2*args.clique_size*args.num_clique)+'_{}'.format(arg_dim[args.dim_reduce])+'.csv',"/Users/suyuhao/Documents/AD/gae_pytorch/output/{}".format(args.dataset))
#"/home/dingj/su000088/computervisionrobustness/3_cnn/"
    '''
    print()
    print("accuracy", "{:.5f}".format(accuracy))
    print("auc", "{:.5f}".format(auc))
    print("f1_score", "{:.5f}".format(f1))    
    print("Job finished!")
    
  
if __name__ == '__main__':
    gae_ad(args)
