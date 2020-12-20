#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:05:59 2020

@author: suyuhao
"""

import argparse
import time
import torch
import os
import shutil

import pandas as pd
import numpy as np
from torch import optim
from model import GCNModelTwoDecodersVAE
from utils import preprocess_graph, make_ad_dataset_both_anomaly, pred_anomaly, precision
from sklearn.metrics import roc_auc_score, f1_score
from optimizer import TwoDecodersVAELoss


parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=int, default=0, help='0 for feature anomaly only, \
                                                    1 for structure only, 2 for all anomaly.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=float, default=0.5, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=float, default=0.25, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate (1 - keep probability).')
parser.add_argument('--clique_size', type=int, default=20, help='create anomaly')
parser.add_argument('--num_clique', type=int, default=15, help='create anomaly')
parser.add_argument('--k', type=int, default=10, help='compare for feature anomaly')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--alpha', type=float, default=1, help='weight parameter for feature error')
parser.add_argument('--beta', type=float, default=1, help='weight parameter for structure error')
parser.add_argument('--gamma', type=float, default=0.2, help='Gamma for the leaky_relu.')
parser.add_argument('--cuda', action='store_true', help='enables cuda')

args = parser.parse_args()


args.device = None
if not args.cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


def gae_ad(args):
    # initialize
    anomaly_type = ["FeatureAnomaly", "StructureAnomaly", "BothAnomaly"]
    print()
    print()
    print()
    print("Initializing normal twodecoders")
    print(args)
    seed = np.random.randint(1000)
    torch.manual_seed(seed)
    print("random seed:", seed)
    loss_plot = []
    auc_plot = []
    f1_plot = []
    accuracy_plot = []
    accuracy_f_plot = []
    accuracy_s_plot =[]
    feature_reconstruction_loss_plot = []
    structure_reconstruction_loss_plot = []

    # obtain basic info
    ad_data_name = "{}_".format(anomaly_type[args.data_type])+"{}_".format(args.dataset) + "clique_size_{}_".format(args.clique_size) + "num_clique_{}".format(
        args.num_clique)+".npy"
    if not os.path.exists(ad_data_name):
        print("no existing ad data found, create new data with anomaly!")
        adj, features, gnd, gnd_f, gnd_s = make_ad_dataset_both_anomaly(args.dataset, args.clique_size, args.num_clique, args.k)
        ad_data_list = [adj, features, gnd, gnd_f, gnd_s]
        np.save(ad_data_name, ad_data_list)
    else:
        print("Found existing ad data, loading...")
        adj, features, gnd, gnd_f, gnd_s = np.load(ad_data_name, allow_pickle=True)
    origin_features = torch.FloatTensor(features).to(args.device)
    gnd = torch.Tensor(gnd).to(args.device)
    gnd_f = torch.Tensor(gnd_f).to(args.device)
    gnd_s = torch.Tensor(gnd_s).to(args.device)
    feat_dim = features.shape[1]
    hidden1 = int(feat_dim * args.hidden1)
    hidden2 = int(feat_dim * args.hidden2)

    # Data preprocessing
    indices, values, shape = preprocess_graph(adj)
    adj_norm = torch.sparse.FloatTensor(indices, values, shape).to_dense().to(args.device)
    print("feature_dim:", feat_dim, "hidden1_dim:", hidden1, "hidden2_dim:", hidden2, "nodes_num:", adj_norm.shape[0],
          "anomaly_num:", 2 * args.clique_size * args.num_clique)
    print()
    print("Start modeling")

    # Start training
    model = GCNModelTwoDecodersVAE(feat_dim, hidden1, hidden2, args.dropout).to(args.device)
    lossFunction = TwoDecodersVAELoss(args.alpha, args.beta)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if torch.cuda.is_available() is True:
        # If using cuda, then need to transfer data from GPU to CPU to save the results.
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            feature_decoder_layer_2, structure_decoder_layer_2 = model(origin_features, adj_norm)
            error, f_error, s_error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss = \
                lossFunction.loss(
                feature_decoder_layer_2,
                structure_decoder_layer_2,
                origin_features, adj_norm)
            total_loss.backward()
            cur_loss = total_loss.item()
            feature_reconstruction_loss = feature_reconstruction_loss.item()
            structure_reconstruction_loss = structure_reconstruction_loss.item()
            optimizer.step()
            loss_plot.append(cur_loss)
            feature_reconstruction_loss_plot.append(feature_reconstruction_loss)
            structure_reconstruction_loss_plot.append(structure_reconstruction_loss)

            auc = roc_auc_score(gnd.cpu().detach().numpy(), error.cpu().detach().numpy())

            pred_gnd = pred_anomaly(error.cpu().detach().numpy(), args.clique_size, args.num_clique, mode=0)
            pred_gnd_s = pred_anomaly(s_error.cpu().detach().numpy(), args.clique_size, args.num_clique, mode=1)
            pred_gnd_f = pred_anomaly(f_error.cpu().detach().numpy(), args.clique_size, args.num_clique, mode=1)
            accuracy = precision(pred_gnd, gnd.cpu().detach().numpy())
            accuracy_f = precision(pred_gnd_f, gnd_f.cpu().detach().numpy())
            accuracy_s = precision(pred_gnd_s, gnd_s.cpu().detach().numpy())

            f1 = f1_score(gnd.cpu().detach().numpy(), pred_gnd)
            accuracy_plot.append(accuracy)
            accuracy_f_plot.append(accuracy_f)
            accuracy_s_plot.append(accuracy_s)
            auc_plot.append(auc)
            f1_plot.append(f1)
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "train_loss=", "{:.5f}".format(cur_loss),
                      "feature_loss=", "{:.5f}".format(feature_reconstruction_loss),
                      "structure_loss=", "{:.5f}".format(structure_reconstruction_loss),
                      "accuracy=", "{:.5f}".format(accuracy),
                      "accuracy_s=", "{:.5f}".format(accuracy_s),
                      "accuracy_f=", "{:.5f}".format(accuracy_f),
                      "time=", "{:.5f}".format(time.time() - t))
    else:
        # CPU case
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            feature_decoder_layer_2, structure_decoder_layer_2 = model(origin_features, adj_norm)
            error, f_error, s_error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss = \
                lossFunction.loss(
                feature_decoder_layer_2,
                structure_decoder_layer_2,
                origin_features, adj_norm)
            total_loss.backward()
            cur_loss = total_loss.item()
            feature_reconstruction_loss = feature_reconstruction_loss.item()
            structure_reconstruction_loss = structure_reconstruction_loss.item()
            optimizer.step()
            loss_plot.append(cur_loss)
            feature_reconstruction_loss_plot.append(feature_reconstruction_loss)
            structure_reconstruction_loss_plot.append(structure_reconstruction_loss)

            auc = roc_auc_score(gnd.detach().numpy(), error.detach().numpy())
            pred_gnd = pred_anomaly(error.detach().numpy(), args.clique_size, args.num_clique, mode=0)
            pred_gnd_s = pred_anomaly(s_error.detach().numpy(), args.clique_size, args.num_clique, mode=1)
            pred_gnd_f = pred_anomaly(f_error.detach().numpy(), args.clique_size, args.num_clique, mode=1)
            accuracy = precision(pred_gnd, gnd.detach().numpy())
            accuracy_f = precision(pred_gnd_f, gnd_f.detach().numpy())
            accuracy_s = precision(pred_gnd_s, gnd_s.detach().numpy())

            f1 = f1_score(gnd.detach().numpy(), pred_gnd)

            accuracy_plot.append(accuracy)
            accuracy_f_plot.append(accuracy_f)
            accuracy_s_plot.append(accuracy_s)
            auc_plot.append(auc)
            f1_plot.append(f1)
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "train_loss=", "{:.5f}".format(cur_loss),
                      "feature_loss=", "{:.5f}".format(feature_reconstruction_loss),
                      "structure_loss=", "{:.5f}".format(structure_reconstruction_loss),
                      "accuracy=", "{:.5f}".format(accuracy),
                      "accuracy_s=", "{:.5f}".format(accuracy_s),
                      "accuracy_f=", "{:.5f}".format(accuracy_f),
                      "time=", "{:.5f}".format(time.time() - t))

    # save the results
    result = {'total_loss': loss_plot,
              'feature_reconstruction_loss': feature_reconstruction_loss_plot,
              'structure_reconstruction_loss': structure_reconstruction_loss_plot,
              'accuracy': accuracy_plot,
              'accuracy_s':accuracy_s_plot,
              'accuracy_f': accuracy_f_plot,
              'auc score': auc_plot,
              'f1 score': f1_plot}
    result_df = pd.DataFrame(data=result)
    result_df.csv_path = 'normal_twodecoders' + \
                         '_{}'.format(args.data_type) + \
                         '_{}'.format(args.dataset) + \
                         '_hidden1_' + '{}'.format(args.hidden1) + \
                         '_hidden2_' + '{}'.format(args.hidden2) + \
                         '_anomaly_{}'.format(2 * args.clique_size * args.num_clique)+'.csv'
    result_df.to_csv(result_df.csv_path)
    if not os.path.exists("/home/augus/ad/gae_pytorch/{}_output".format(args.dataset)):
        os.makedirs('{}_output'.format(args.dataset))
    shutil.move(result_df.csv_path,"/home/augus/ad/gae_pytorch/{}_output".format(args.dataset))
    print()
    print("accuracy", "{:.5f}".format(accuracy))
    print("accuracy_s", "{:.5f}".format(accuracy_s))
    print("accuracy_f", "{:.5f}".format(accuracy_f))
    print("auc", "{:.5f}".format(auc))
    print("f1_score", "{:.5f}".format(f1))
    print("Job finished!")



if __name__ == '__main__':
    gae_ad(args)
