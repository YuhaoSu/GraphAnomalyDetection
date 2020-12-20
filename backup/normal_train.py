#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:05:59 2020

@author: suyuhao
"""

# from __future__ import division
# from __future__ import print_function

import argparse
import time
import torch
import pandas as pd
import numpy as np
import shutil
import os

from torch import optim
from backup.model import GCNModelVAE  # GCNModelScatRSRAE
from backup.utils import  preprocess_graph, make_ad_dataset, pred_anomaly, precision
from sklearn.metrics import roc_auc_score,f1_score #, average_precision_score
from backup.optimizer import ae_loss_function #, scat_rsrae_loss_function

#import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
#parser.add_argument('--seed', type=int, default=100, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=float, default=0.5, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=float, default=0.5, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.004, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
#parser.add_argument('--rsr_dim', type=int, default=100, help='dim of rsr layer')
parser.add_argument('--clique_size', type=int, default=20, help='create anomaly')
parser.add_argument('--num_clique', type=int, default=15, help='create anomaly')
parser.add_argument('--k', type=int, default=10, help='compare for feature anomaly')
#parser.add_argument('--rsr_enable', type=int, default=1, help='whether use rsr, 0 is true ,1 is false')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
#parser.add_argument('--scat_version', type=int, default=0, help='0 for scat, 1 for diffusion')
#parser.add_argument('--dim_reduce', type=int, default=1, help='0 for pca, 1 for fms, 2 for no dim reduction')    
parser.add_argument('--alpha', type=float, default=1, help='weight parameter for feature error')  
parser.add_argument('--beta', type=float, default=1, help='weight parameter for structure error')  
#parser.add_argument('--gamma', type=float, default=1, help='weight parameter for projection error') 
#parser.add_argument('--delta', type=float, default=1, help='weight parameter for pca error')
parser.add_argument('--cuda', action='store_true', help='enables cuda')


args = parser.parse_args()

args.device = None
if not args.cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
   

def gae_ad(args):
    #initialize
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
    features = torch.FloatTensor(features).to(args.device)
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
    
    #Start training
    model = GCNModelVAE(feat_dim, hidden1, hidden2, args.dropout).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if torch.cuda.is_available() is True: 
        #If using cuda, then need to transfer data from GPU to CPU to save the results.
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            feature_decoder_layer_2, structure_decoder_layer_2 = model(features, adj_norm)
            error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss = ae_loss_function(feature_decoder_layer_2,
                                                                                                             structure_decoder_layer_2, 
                                                                                                             features, adj_norm, args.alpha,
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
            feature_decoder_layer_2, structure_decoder_layer_2 = model(features, adj_norm)
            error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss = ae_loss_function(feature_decoder_layer_2,
                                                                                                             structure_decoder_layer_2, 
                                                                                                             features, adj_norm, args.alpha,
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
    result_df.csv_path = 'normal_dataset_'+'{}'.format(args.dataset)+'_hidden2_'+'{}'.format(args.hidden2)+'_anomaly_{}'.format(2*args.clique_size*args.num_clique)+'.csv'
    result_df.to_csv(result_df.csv_path)
    if not os.path.exists("/Users/suyuhao/Documents/AD/gae_pytorch/output/{}".format(args.dataset)):
        os.makedirs('{}'.format(args.dataset))
    shutil.move('normal_dataset_'+'{}'.format(args.dataset)+'_hidden2_'+'{}'.format(args.hidden2)+'_anomaly_{}'.format(2*args.clique_size*args.num_clique)+'.csv',"/Users/suyuhao/Documents/AD/gae_pytorch/output/{}".format(args.dataset))
    
    print()
    print("accuracy", "{:.5f}".format(accuracy))
    print("auc", "{:.5f}".format(auc))
    print("f1_score", "{:.5f}".format(f1))    
    print("Job finished!")
    
  
if __name__ == '__main__':
    gae_ad(args)
