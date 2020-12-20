from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import torch
from torch import optim
import matplotlib.pyplot as plt
from backup.model import GCNModelVAE,GCNModelRSRAE
from backup.utils import  preprocess_graph, make_ad_dataset, pred_anomaly, precision
from sklearn.metrics import roc_auc_score, f1_score #, average_precision_score
from backup.optimizer import ae_loss_function, rsrae_loss_function


parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
#parser.add_argument('--seed', type=int, default=100, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=float, default=0.75, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=float, default=0.5, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.004, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
#parser.add_argument('--rsr_dim', type=int, default=20, help='dim of rsr layer')
parser.add_argument('--clique_size', type=int, default=20, help='create anomaly')
parser.add_argument('--num_clique', type=int, default=20, help='create anomaly')
parser.add_argument('--k', type=int, default=10, help='compare for feature anomaly')
parser.add_argument('--rsr_enable', type=int, default=1, help='whether use rsr, 0 is true ,1 is false')
parser.add_argument('--dataset', type=str, default='cora', help='type of dataset.')
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
    arg_rsr = [True, False]
    print()
    print(args)
    seed = np.random.randint(1000)
    torch.manual_seed(seed)
    print("seed:",seed)
    print("start modeling")
    
    loss_plot = []
    epoch_plot = []
    auc_plot = []
    f1_plot = []
    
    adj, features, gnd = make_ad_dataset(args.dataset, args.clique_size, args.num_clique, args.k)
    features = torch.FloatTensor(features).to(args.device)
    gnd = torch.Tensor(gnd).to(args.device)
    feat_dim = features.shape[1]
    hidden1 = int(feat_dim * args.hidden1)
    hidden2 = int(feat_dim * args.hidden2)
    # Some preprocessing
    indices, values, shape = preprocess_graph(adj)
    adj_norm = torch.sparse.FloatTensor(indices, values, shape).to(args.device)
    print("feature_dim:",feat_dim, "hidden1:",hidden1, "hidden2:",hidden2,"adj_norm_dim:",adj_norm.shape[0])


    if arg_rsr[args.rsr_enable] is False:
        print("using model without RSR")
        feature_reconstruction_loss_plot = []
        structure_reconstruction_loss_plot = []
        
        model = GCNModelVAE(feat_dim, hidden1, hidden2, args.dropout).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
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
            #print(cur_loss)
            feature_reconstruction_loss = feature_reconstruction_loss.item()
            structure_reconstruction_loss = structure_reconstruction_loss.item()
            optimizer.step()
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "time=", "{:.5f}".format(time.time() - t))
                
            loss_plot.append(cur_loss)
            feature_reconstruction_loss_plot.append(feature_reconstruction_loss)
            structure_reconstruction_loss_plot.append(structure_reconstruction_loss)
            epoch_plot.append(epoch)
            
            if torch.cuda.is_available() is True:
                auc = roc_auc_score(gnd.cpu().detach().numpy(), error.cpu().detach().numpy())
                pred_gnd = pred_anomaly(error.cpu().detach().numpy(), args.clique_size, args.num_clique)
                f1 = f1_score(gnd.cpu().detach().numpy(), pred_gnd)
            else: 
                auc = roc_auc_score(gnd.detach().numpy(), error.detach().numpy())
                pred_gnd = pred_anomaly(error.detach().numpy(), args.clique_size, args.num_clique)
                f1 = f1_score(gnd.detach().numpy(), pred_gnd)
            auc_plot.append(auc)
            f1_plot.append(f1)

        plt.plot(np.array(epoch_plot),np.array(feature_reconstruction_loss_plot), 'r--', label = 'feature_reconstruction_loss')
        plt.plot(np.array(epoch_plot),np.array(structure_reconstruction_loss_plot), 'g--', label = 'structure_reconstruction_loss')
        plt.xlabel('epoch')
        plt.title('Loss plot by each')
        plt.legend()
        plt.show()
        plt.savefig("loss_by_each_"+"{}".format(args.dataset)+"hidden2"+"{}".format(args.hidden2)+"_lr_"+"{}".format(args.lr)+"clique_size_"+"{}".format(args.clique_size)+".jpg")
        plt.close() 
        
        plt.plot(np.array(epoch_plot),np.array(loss_plot), 'b--', label = 'total_loss')
        plt.xlabel('epoch')
        plt.title('Total loss plot')
        plt.legend()
        plt.show()
        plt.savefig("total_loss_"+"{}".format(args.dataset)+"hidden2"+"{}".format(args.hidden2)+"_lr_"+"{}".format(args.lr)+"clique_size_"+"{}".format(args.clique_size)+".jpg")
        plt.close() 

        plt.plot(np.array(epoch_plot),np.array(auc_plot),'b--', label = 'auc_plot')
        plt.xlabel('epoch')
        plt.title('AUC plot')
        plt.legend()
        plt.show()            
        plt.savefig("auc_"+"{}".format(args.dataset)+"hidden2"+"{}".format(args.hidden2)+"_lr_"+"{}".format(args.lr)+"clique_size_"+"{}".format(args.clique_size)+".jpg")
        plt.close() 
        
        plt.plot(np.array(epoch_plot),np.array(f1_plot),'b--',label = 'f1' )
        plt.xlabel('epoch')
        plt.title('F1 plot')
        plt.legend()
        plt.show()
        plt.savefig("f1_"+"{}".format(args.dataset)+"hidden2"+"{}".format(args.hidden2)+"_lr_"+"{}".format(args.lr)+"clique_size_"+"{}".format(args.clique_size)+".jpg")
        plt.close()  
        
    else:
        print("using model with RSR")
        feature_reconstruction_loss_plot = []
        structure_reconstruction_loss_plot = []
        proj_loss_plot = []
        pca_loss_plot = []
        model = GCNModelRSRAE(feat_dim, hidden1, hidden2, args.dropout, args.rsr_dim).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.epochs):
            
            t = time.time()
            model.train()
            optimizer.zero_grad()
            encoder_layer_2, rsrlayer, rsr_output, feature_decoder_layer_2, structure_decoder_layer_2 = model(features, adj_norm, args.rsr_dim)
            error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss, proj_loss, pca_loss = rsrae_loss_function(args.device, encoder_layer_2, rsrlayer, 
                                                                                                                                     rsr_output, feature_decoder_layer_2, 
                                                                                                                                     structure_decoder_layer_2, features, 
                                                                                                                                     adj_norm, args.rsr_dim, args.alpha, 
                                                                                                                                     args.beta, args.gamma, args.delta)

            total_loss.backward()
            cur_loss = total_loss.item()
            feature_reconstruction_loss = feature_reconstruction_loss.item()
            structure_reconstruction_loss = structure_reconstruction_loss.item()
            proj_loss = proj_loss.item()
            pca_loss = pca_loss.item()
            
            feature_reconstruction_loss_plot.append(feature_reconstruction_loss)
            structure_reconstruction_loss_plot.append(structure_reconstruction_loss)
            proj_loss_plot.append(proj_loss)
            pca_loss_plot.append(pca_loss)
            
            optimizer.step()
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                      "time=", "{:.5f}".format(time.time() - t))
            
            loss_plot.append(cur_loss)
            epoch_plot.append(epoch)
            
            
            if torch.cuda.is_available() is True:

                auc = roc_auc_score(gnd.cpu().detach().numpy(), error.cpu().detach().numpy())
                pred_gnd = pred_anomaly(error.cpu().detach().numpy(), args.clique_size, args.num_clique)
                f1 = f1_score(gnd.cpu().detach().numpy(), pred_gnd)
                
            else: 
                auc = roc_auc_score(gnd.detach().numpy(), error.detach().numpy())
                pred_gnd = pred_anomaly(error.detach().numpy(), args.clique_size, args.num_clique)
                f1 = f1_score(gnd.detach().numpy(), pred_gnd)
            #accuracy = precision(pred_gnd, gnd, args.clique_size, args.num_clique)
            #print("accuracy", "{:.5f}".format(accuracy))
            auc_plot.append(auc)
            f1_plot.append(f1)
                
        plt.plot(np.array(epoch_plot),np.array(feature_reconstruction_loss_plot),label = 'feature_reconstruction_loss')
        plt.plot(np.array(epoch_plot),np.array(structure_reconstruction_loss_plot), label = 'structure_reconstruction_loss')
        plt.plot(np.array(epoch_plot),np.array(proj_loss_plot), label = 'proj_loss')
        plt.plot(np.array(epoch_plot),np.array(pca_loss_plot), label = 'pca_loss')
        plt.xlabel('epoch')
        plt.title('Loss plot by each')
        plt.legend()
        plt.show()
        plt.savefig("loss_by_each_"+"{}".format(args.dataset)+"_rsr_enable_true_alpha_"+"{}".format(args.alpha)+"_beta_"+"{}".format(args.beta)+"_gamma_"+"{}".format(args.gamma)+"_delta_"+"{}".format(args.delta)+".jpg")
        plt.close()

        plt.plot(np.array(epoch_plot),np.array(loss_plot), label = 'loss_plot')
        plt.legend()
        plt.show()
        plt.savefig("total_loss_"+"{}".format(args.dataset)+"_rsr_enable_true_alpha_"+"{}".format(args.alpha)+"_beta_"+"{}".format(args.beta)+"_gamma_"+"{}".format(args.gamma)+"_delta_"+"{}".format(args.delta)+".jpg")
        plt.close() 

        plt.plot(np.array(epoch_plot),np.array(auc_plot),'b--', label = 'auc_plot')
        plt.legend()
        plt.show()
        plt.savefig("auc_"+"{}".format(args.dataset)+"_rsr_enable_true_alpha_"+"{}".format(args.alpha)+"_beta_"+"{}".format(args.beta)+"_gamma_"+"{}".format(args.gamma)+"_delta_"+"{}".format(args.delta)+".jpg")
        plt.close() 
        
        plt.plot(np.array(epoch_plot),np.array(f1_plot),'b--', label = 'f1_plot')
        plt.legend()
        plt.show()
        plt.savefig("f1_"+"{}".format(args.dataset)+"_rsr_enable_true_alpha_"+"{}".format(args.alpha)+"_beta_"+"{}".format(args.beta)+"_gamma_"+"{}".format(args.gamma)+"_delta_"+"{}".format(args.delta)+".jpg")
        plt.close() 
    
    
    if torch.cuda.is_available() is True:
        
        auc = roc_auc_score(gnd.cpu().detach().numpy(), error.cpu().detach().numpy())
        pred_gnd = pred_anomaly(error.cpu().detach().numpy(), args.clique_size, args.num_clique)
        accuracy = precision(pred_gnd, gnd.cpu().detach().numpy(), args.clique_size, args.num_clique)
        f1 = f1_score(gnd.cpu().detach().numpy(), pred_gnd)
    else: 
        auc = roc_auc_score(gnd.detach().numpy(), error.detach().numpy())
        pred_gnd = pred_anomaly(error.detach().numpy(), args.clique_size, args.num_clique)
        accuracy = precision(pred_gnd, gnd.detach().numpy(), args.clique_size, args.num_clique)
        f1 = f1_score(gnd.detach().numpy(), pred_gnd)
    print("accuracy", "{:.5f}".format(accuracy))
    
    print("f1_score", "{:.5f}".format(f1))    
    print("Optimization Finished!")
  

if __name__ == '__main__':
    gae_ad(args)
