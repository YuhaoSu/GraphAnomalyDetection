import torch
import torch.nn.modules.loss


def ae_loss_function(feature_pred, structure_pred, features, adj_norm, alpha, beta):
    feature_reconstruction_error = torch.norm(feature_pred - features, p=2, dim=1)
    structure_reconstruction_error = torch.norm(structure_pred - adj_norm, p=2, dim=1)

    error = alpha * feature_reconstruction_error + beta * structure_reconstruction_error
    
    feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
    structure_reconstruction_loss = torch.mean(structure_reconstruction_error)
    
    total_loss = alpha * feature_reconstruction_loss + beta * structure_reconstruction_loss
    #total_loss = structure_reconstruction_loss
    #print(feature_reconstruction_loss)
    #print(structure_reconstruction_loss)
    #print(total_loss)
    #print()
    return error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss


def rsrae_loss_function(device, encoder_layer_2, rsrlayer, rsr_output, feature_decoder_layer_2, 
                        structure_decoder_layer_2, features, adj_norm, rsr_dim,
                        alpha, beta, gamma, delta):
    
    feature_reconstruction_error = torch.norm(feature_decoder_layer_2 -features,p=2, dim=1)
    structure_reconstruction_error = torch.norm(structure_decoder_layer_2 -adj_norm,p=2, dim=1)
    error = alpha * feature_reconstruction_error + beta * structure_reconstruction_error
    
    feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
    structure_reconstruction_loss = torch.mean(structure_reconstruction_error)
    
    proj_loss = torch.mean((torch.mm(torch.t(rsrlayer), rsrlayer) - torch.eye(rsr_dim).to(device)).pow(2))
    temp = torch.mm(rsr_output, torch.t(rsrlayer))
    pca_loss = torch.mean(torch.norm(encoder_layer_2 - temp, p=2, dim =1))
   
    total_loss = alpha * feature_reconstruction_loss + beta * structure_reconstruction_loss + gamma * proj_loss + delta * pca_loss
        
    #print(feature_reconstruction_loss)
    #print(structure_reconstruction_loss)
    #print(proj_loss)
    #print(pca_loss)
    #print()
    
    return error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss, proj_loss, pca_loss



def scat_ae_loss_function(feature_pred, structure_pred, features, adj_norm, alpha, beta):
    feature_reconstruction_error = torch.norm(feature_pred - features, p=2, dim=1)
    structure_reconstruction_error = torch.norm(structure_pred - adj_norm, p=2, dim=1)

    error = alpha * feature_reconstruction_error + beta * structure_reconstruction_error
    
    feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
    structure_reconstruction_loss = torch.mean(structure_reconstruction_error)
    
    total_loss = alpha * feature_reconstruction_loss + beta * structure_reconstruction_loss
    #total_loss = structure_reconstruction_loss
    #print(feature_reconstruction_loss)
    #print(structure_reconstruction_loss)
    #print(total_loss)
    #print()
    return error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss

def scat_rsrae_loss_function(device, y_features, rsrlayer, rsr_output, feature_decoder_layer_2, 
                        structure_decoder_layer_2, features, adj_norm, rsr_dim,
                        alpha, beta, gamma, delta):
    
    feature_reconstruction_error = torch.norm(feature_decoder_layer_2 -features,p=2, dim=1)
    structure_reconstruction_error = torch.norm(structure_decoder_layer_2 -adj_norm,p=2, dim=1)
    error = alpha * feature_reconstruction_error + beta * structure_reconstruction_error
    
    feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
    structure_reconstruction_loss = torch.mean(structure_reconstruction_error)
    
    proj_loss = torch.mean((torch.mm(torch.t(rsrlayer), rsrlayer) - torch.eye(rsr_dim).to(device)).pow(2))
    temp = torch.mm(rsr_output, torch.t(rsrlayer))
    pca_loss = torch.mean(torch.norm(y_features - temp, p=2, dim =1))
   
    total_loss = alpha * feature_reconstruction_loss + beta * structure_reconstruction_loss + gamma * proj_loss + delta * pca_loss
        
    #print(feature_reconstruction_loss)
    #print(structure_reconstruction_loss)
    #print(proj_loss)
    #print(pca_loss)
    #print()
    
    return error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss, proj_loss, pca_loss