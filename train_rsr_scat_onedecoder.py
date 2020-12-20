#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:05:59 2020

@author: suyuhao
"""
import argparse
import time
import torch
import scat
import os
import shutil
import networkx as nx
import pandas as pd
import numpy as np
from torch import optim
from model import GCNModelRsrScatFeatureOnlyVAE, GCNModelRsrScatStructureOnlyVAE
from utils import preprocess_graph, make_ad_dataset_both_anomaly, make_ad_dataset_structure_anomaly,\
    make_ad_dataset_feature_anomaly, pred_anomaly, precision, dim_reduction
from sklearn.metrics import roc_auc_score, f1_score
from optimizer import RsrFeatureOnlyVAELoss, RsrStructureOnlyVAELoss
from fms import FMS

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=int, default=1, help='0 for feature anomaly only, \
                                                    1 for structure only, 2 for all anomaly.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=float, default=0.5, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=float, default=0.25, help='Number of units in hidden layer 2.')
parser.add_argument('--rsr_dim', type=float, default=0.1, help='Number of units in rsr layer.')
parser.add_argument('--att', type=int, default=0, help='0 for no attention, 1 for all attention, \
                            2 for feature attention, 3 for structure, so far only support no attention.')
parser.add_argument('--dim_reduce', type=int, default=0, help='0 for pca, 1 for fms, 2 for no dim reduction')
parser.add_argument('--decoder', type=int, default=1, help='0 for feature only, 1 for structure only.')
parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate (1 - keep probability).')
parser.add_argument('--clique_size', type=int, default=20, help='create anomaly')
parser.add_argument('--num_clique', type=int, default=15, help='create anomaly')
parser.add_argument('--k', type=int, default=10, help='compare for feature anomaly')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
parser.add_argument('--alpha', type=float, default=1, help='weight parameter for feature error')
parser.add_argument('--beta', type=float, default=1, help='weight parameter for structure error')
parser.add_argument('--gamma', type=float, default=1, help='Gamma for the projection error in Rsr layer.')
parser.add_argument('--delta', type=float, default=1, help='Gamma for the pca error in Rsr layer.')
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
    decoder_name = ["FeatureDecoder", "StructureDecoder"]
    arg_dim = ['PCA', 'FMS', 'NoReduction']
    att = ["NoAtt", "DoubleAtt", "FeaatureAtt", "StructureAtt"]
    print()
    print()
    print()
    print("Initializing rsr scat onedecoder training")
    print(args)
    seed = np.random.randint(1000)
    torch.manual_seed(seed)
    print("random seed:", seed)
    total_loss_plot = []
    reconstruction_loss_plot = []
    proj_loss_plot = []
    pca_loss_plot = []
    auc_plot = []
    f1_plot = []
    accuracy_plot = []


    # obtain basic info
    ad_data_name = "{}_".format(anomaly_type[args.data_type]) + "{}_".format(args.dataset) + "clique_size_{}_".format(
        args.clique_size) + "num_clique_{}".format(
        args.num_clique) + ".npy"
    if not os.path.exists(ad_data_name):
        print("no existing ad data found, create new data with anomaly!")
        if args.data_type == 0:
            adj, features, gnd = make_ad_dataset_feature_anomaly(args.dataset, args.clique_size, args.num_clique,
                                                                 args.k)
            ad_data_list = [adj, features, gnd]
            np.save(ad_data_name, ad_data_list)
            origin_features = torch.FloatTensor(features).to(args.device)
            gnd = torch.Tensor(gnd).to(args.device)
        elif args.data_type == 1:
            adj, features, gnd = make_ad_dataset_structure_anomaly(args.dataset, args.clique_size, args.num_clique,
                                                                   args.k)
            ad_data_list = [adj, features, gnd]
            np.save(ad_data_name, ad_data_list)
            origin_features = torch.FloatTensor(features).to(args.device)
            gnd = torch.Tensor(gnd).to(args.device)
        elif args.data_type == 2:
            adj, features, gnd, gnd_f, gnd_s = make_ad_dataset_both_anomaly(args.dataset, args.clique_size,
                                                                           args.num_clique, args.k)
            ad_data_list = [adj, features, gnd, gnd_f, gnd_s]
            np.save(ad_data_name, ad_data_list)
            origin_features = torch.FloatTensor(features).to(args.device)
            gnd = torch.Tensor(gnd).to(args.device)
            gnd_f = torch.Tensor(gnd_f).to(args.device)
            gnd_s = torch.Tensor(gnd_s).to(args.device)
        else:
            raise Exception("No valid data! Try to create another different data type!")
    else:
        if args.data_type == 0:
            print("Found existing ad data, loading...")
            adj, features, gnd = np.load(ad_data_name, allow_pickle=True)
            origin_features = torch.FloatTensor(features).to(args.device)
            gnd = torch.Tensor(gnd).to(args.device)
        elif args.data_type == 1:
            print("Found existing ad data, loading...")
            adj, features, gnd = np.load(ad_data_name, allow_pickle=True)
            origin_features = torch.FloatTensor(features).to(args.device)
            gnd = torch.Tensor(gnd).to(args.device)
        elif args.data_type == 2:
            print("Found existing ad data, loading...")
            adj, features, gnd, gnd_f, gnd_s = np.load(ad_data_name, allow_pickle=True)
            origin_features = torch.FloatTensor(features).to(args.device)
            gnd = torch.Tensor(gnd).to(args.device)
            gnd_f = torch.Tensor(gnd_f).to(args.device)
            gnd_s = torch.Tensor(gnd_s).to(args.device)
        else:
            raise Exception("No valid data! Try to load another different data type!")


    feat_dim = features.shape[1]
    hidden1 = int(feat_dim * args.hidden1)
    hidden2 = int(feat_dim * args.hidden2)
    rsr_dim = int(feat_dim * args.rsr_dim)

    # Data preprocessing
    indices, values, shape = preprocess_graph(adj)
    adj_norm = torch.sparse.FloatTensor(indices, values, shape).to_dense().to(args.device)
    print("feature_dim:", feat_dim, "hidden1_dim:", hidden1, "hidden2_dim:", hidden2, "rsr_dim:", rsr_dim, \
          "nodes_num:", adj_norm.shape[0],"anomaly_num:", 2 * args.clique_size * args.num_clique)
    print()
    print("Start modeling")
    print("using wavelet scattering transform")

    A = adj
    L = nx.linalg.laplacianmatrix.laplacian_matrix(nx.from_scipy_sparse_matrix(A))
    lamb, V = np.linalg.eigh(L.toarray())
    y_features = scat.getRep(features, lamb, V, layer=3)
    print("y_features shape after scatting", y_features.shape, type(y_features))

    if arg_dim[args.dim_reduce] == 'PCA':
        print("using PCA to reduce dim")
        features = dim_reduction(y_features, num_of_components=hidden2)
    elif arg_dim[args.dim_reduce] == 'FMS':
        print("using FMS to reduce dim")
        fms_ = FMS(n_components=hidden2)
        features = fms_.fit_transform(y_features)
    elif arg_dim[args.dim_reduce] == 'NoReduction':
        hidden2 = y_features.shape[1]
        features = y_features
        print(hidden2)
    else:
        raise Exception("No valid option in arg_dim, check input!")
    features = torch.FloatTensor(features).to(args.device)
    print("y_features shape after " + '{}'.format(arg_dim[args.dim_reduce]), features.shape, type(features))

    # Start training
    if args.decoder == 0: # feature decoder only
        lossFunction = RsrFeatureOnlyVAELoss(args.alpha, args.beta, args.gamma, args.delta, rsr_dim, args.device)
        if args.att == 0: # no att
            model = GCNModelRsrScatFeatureOnlyVAE(feat_dim, hidden1, hidden2, args.dropout, rsr_dim).to(args.device)
        else:
            raise Exception("No valid option for model, check input!")

    elif args.decoder == 1: # structure decoder only
        lossFunction = RsrStructureOnlyVAELoss(args.alpha, args.beta, args.gamma, args.delta, rsr_dim, args.device)
        if args.att == 0: # no att
            model = GCNModelRsrScatStructureOnlyVAE(feat_dim, hidden1, hidden2, args.dropout, rsr_dim).to(args.device)
        else:
            raise Exception("No valid option for model, check input!")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if torch.cuda.is_available() is True:
        # If using cuda, then need to transfer data from GPU to CPU to save the results.
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            decoder_layer_2, rsrlayer, rsr_output = model(features, adj_norm)
            error, total_loss, reconstruction_loss, proj_loss, pca_loss = \
                lossFunction.loss(decoder_layer_2, origin_features, adj_norm, features, rsrlayer, rsr_output)

            total_loss.backward()
            cur_loss = total_loss.item()
            reconstruction_loss = reconstruction_loss.item()
            proj_loss = proj_loss.item()
            pca_loss = pca_loss.item()
            optimizer.step()
            total_loss_plot.append(cur_loss)
            reconstruction_loss_plot.append(reconstruction_loss)
            proj_loss_plot.append(proj_loss)
            pca_loss_plot.append(pca_loss)

            auc = roc_auc_score(gnd.cpu().detach().numpy(), error.cpu().detach().numpy())
            pred_gnd = pred_anomaly(error.cpu().detach().numpy(), args.clique_size, args.num_clique, mode=0)
            accuracy = precision(pred_gnd, gnd.cpu().detach().numpy())

            f1 = f1_score(gnd.cpu().detach().numpy(), pred_gnd)
            accuracy_plot.append(accuracy)

            auc_plot.append(auc)
            f1_plot.append(f1)
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "train_loss=", "{:.5f}".format(cur_loss),
                      "reconstruction_loss=", "{:.5f}".format(reconstruction_loss),
                      "proj_loss=", "{:.5f}".format(proj_loss),
                      "pca_loss=", "{:.5f}".format(pca_loss),
                      "accuracy=", "{:.5f}".format(accuracy),
                      "time=", "{:.5f}".format(time.time() - t))
    else:
        # CPU case
        for epoch in range(args.epochs):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            decoder_layer_2, rsrlayer, rsr_output = model(features, adj_norm)
            error, total_loss, reconstruction_loss, proj_loss, pca_loss = \
                lossFunction.loss(decoder_layer_2, origin_features, adj_norm, features, rsrlayer, rsr_output)
            total_loss.backward()
            cur_loss = total_loss.item()
            reconstruction_loss = reconstruction_loss.item()
            proj_loss = proj_loss.item()
            pca_loss = pca_loss.item()
            optimizer.step()
            total_loss_plot.append(cur_loss)
            reconstruction_loss_plot.append(reconstruction_loss)
            proj_loss_plot.append(proj_loss)
            pca_loss_plot.append(pca_loss)

            auc = roc_auc_score(gnd.detach().numpy(), error.detach().numpy())
            pred_gnd = pred_anomaly(error.detach().numpy(), args.clique_size, args.num_clique, mode=0)
            accuracy = precision(pred_gnd, gnd.detach().numpy())
            f1 = f1_score(gnd.detach().numpy(), pred_gnd)

            accuracy_plot.append(accuracy)

            auc_plot.append(auc)
            f1_plot.append(f1)
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "train_loss=", "{:.5f}".format(cur_loss),
                      "reconstruction_loss=", "{:.5f}".format(reconstruction_loss),
                      "proj_loss=", "{:.5f}".format(proj_loss),
                      "pca_loss=", "{:.5f}".format(pca_loss),
                      "accuracy=", "{:.5f}".format(accuracy),
                      "time=", "{:.5f}".format(time.time() - t))

    # save the results
    result = {'total_loss': total_loss_plot,
              'reconstruction_loss': reconstruction_loss_plot,
              'proj_loss': proj_loss_plot,
              'pca_loss': pca_loss_plot,
              'accuracy': accuracy_plot,
              'auc score': auc_plot,
              'f1 score': f1_plot}
    result_df = pd.DataFrame(data=result)
    result_df.csv_path = 'rsr_scat_onedecoder' + \
                         '_{}'.format(args.data_type) + \
                         '_{}'.format(args.dataset) + \
                         '_hidden1_' + '{}'.format(args.hidden1) + \
                         '_hidden2_' + '{}'.format(args.hidden2) + \
                         '_rsr_dim' + '{}'.format(args.rsr_dim) + \
                         '_'+'{}'.format(decoder_name[args.decoder])+ \
                         '_{}'.format(arg_dim[args.dim_reduce]) + \
                         '_anomaly_{}'.format(2 * args.clique_size * args.num_clique)+'.csv'
    result_df.to_csv(result_df.csv_path)
    if not os.path.exists("/home/augus/ad/gae_pytorch/{}_output".format(args.dataset)):
        os.makedirs('{}_output'.format(args.dataset))
    shutil.move(result_df.csv_path,"/home/augus/ad/gae_pytorch/{}_output".format(args.dataset))
    print()
    print("accuracy", "{:.5f}".format(accuracy))
    print("auc", "{:.5f}".format(auc))
    print("f1_score", "{:.5f}".format(f1))
    print("Job finished!")


if __name__ == '__main__':
    gae_ad(args)

