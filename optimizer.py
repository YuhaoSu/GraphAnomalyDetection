import torch
import torch.nn



class TwoDecodersVAELoss:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def loss(self, feature_pred, structure_pred, features, adj_norm):
        feature_reconstruction_error = torch.norm(feature_pred - features, p=2, dim=1)
        structure_reconstruction_error = torch.norm(structure_pred - adj_norm, p=2, dim=1)
        error = self.alpha * feature_reconstruction_error + self.beta * structure_reconstruction_error
        feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
        structure_reconstruction_loss = torch.mean(structure_reconstruction_error)
        total_loss = self.alpha * feature_reconstruction_loss + self.beta * structure_reconstruction_loss
        return error, feature_reconstruction_error, structure_reconstruction_error, total_loss, \
               feature_reconstruction_loss, structure_reconstruction_loss




class FeatureOnlyVAELoss:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def loss(self, feature_pred, features, adj_norm):
        feature_reconstruction_error = torch.norm(feature_pred - features, p=2, dim=1)
        feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
        return feature_reconstruction_error, feature_reconstruction_loss


class StructureOnlyVAELoss:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def loss(self, structure_pred, features, adj_norm):
        structure_reconstruction_error = torch.norm(structure_pred - adj_norm, p=2, dim=1)
        structure_reconstruction_loss = torch.mean(structure_reconstruction_error)
        return structure_reconstruction_error, structure_reconstruction_loss

class RsrTwoDecodersVAELoss:
    def __init__(self, alpha, beta, gamma, delta, rsr_dim, device):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rsr_dim = rsr_dim
        self.device = device

    def loss(self, feature_pred, structure_pred, features, adj_norm, encoder_layer_2, rsrlayer, rsr_output):
        feature_reconstruction_error = torch.norm(feature_pred - features, p=2, dim=1)
        structure_reconstruction_error = torch.norm(structure_pred - adj_norm, p=2, dim=1)
        error = self.alpha * feature_reconstruction_error + self.beta * structure_reconstruction_error

        feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
        structure_reconstruction_loss = torch.mean(structure_reconstruction_error)

        proj_loss = torch.mean((torch.mm(torch.t(rsrlayer), rsrlayer) - torch.eye(self.rsr_dim).to(self.device)).pow(2))
        temp = torch.mm(rsr_output, torch.t(rsrlayer))
        pca_loss = torch.mean(torch.norm(encoder_layer_2 - temp, p=2, dim=1))

        total_loss = self.alpha * feature_reconstruction_loss + self.beta * structure_reconstruction_loss + \
                     self.gamma * proj_loss + self.delta * pca_loss

        return error, feature_reconstruction_error, structure_reconstruction_error, \
               total_loss, feature_reconstruction_loss, structure_reconstruction_loss, \
               proj_loss, pca_loss

class RsrFeatureOnlyVAELoss:
    def __init__(self, alpha, beta, gamma, delta, rsr_dim, device):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rsr_dim = rsr_dim
        self.device = device

    def loss(self, feature_pred, features, adj_norm, encoder_layer_2, rsrlayer, rsr_output):
        feature_reconstruction_error = torch.norm(feature_pred - features, p=2, dim=1)
        feature_reconstruction_loss = torch.mean(feature_reconstruction_error)

        proj_loss = torch.mean((torch.mm(torch.t(rsrlayer), rsrlayer) - torch.eye(self.rsr_dim).to(self.device)).pow(2))
        temp = torch.mm(rsr_output, torch.t(rsrlayer))
        pca_loss = torch.mean(torch.norm(encoder_layer_2 - temp, p=2, dim=1))

        total_loss = self.alpha * feature_reconstruction_loss + \
                     self.gamma * proj_loss + self.delta * pca_loss

        return feature_reconstruction_error, total_loss, feature_reconstruction_loss, proj_loss, pca_loss


class RsrStructureOnlyVAELoss:
    def __init__(self, alpha, beta, gamma, delta, rsr_dim, device):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.rsr_dim = rsr_dim
        self.device = device

    def loss(self, structure_pred, features, adj_norm, encoder_layer_2, rsrlayer, rsr_output):
        structure_reconstruction_error = torch.norm(structure_pred - adj_norm, p=2, dim=1)
        structure_reconstruction_loss = torch.mean(structure_reconstruction_error)

        proj_loss = torch.mean((torch.mm(torch.t(rsrlayer), rsrlayer) - torch.eye(self.rsr_dim).to(self.device)).pow(2))
        temp = torch.mm(rsr_output, torch.t(rsrlayer))
        pca_loss = torch.mean(torch.norm(encoder_layer_2 - temp, p=2, dim=1))

        total_loss = self.beta * structure_reconstruction_loss + \
                     self.gamma * proj_loss + self.delta * pca_loss

        return structure_reconstruction_error, total_loss, structure_reconstruction_loss, proj_loss, pca_loss


# def rsrae_loss_function(device, encoder_layer_2, rsrlayer, rsr_output, feature_decoder_layer_2,
#                         structure_decoder_layer_2, features, adj_norm, rsr_dim,
#                         alpha, beta, gamma, delta):
#     feature_reconstruction_error = torch.norm(feature_decoder_layer_2 - features, p=2, dim=1)
#     structure_reconstruction_error = torch.norm(structure_decoder_layer_2 - adj_norm, p=2, dim=1)
#     error = alpha * feature_reconstruction_error + beta * structure_reconstruction_error
#
#     feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
#     structure_reconstruction_loss = torch.mean(structure_reconstruction_error)
#
#     proj_loss = torch.mean((torch.mm(torch.t(rsrlayer), rsrlayer) - torch.eye(rsr_dim).to(device)).pow(2))
#     temp = torch.mm(rsr_output, torch.t(rsrlayer))
#     pca_loss = torch.mean(torch.norm(encoder_layer_2 - temp, p=2, dim=1))
#
#     total_loss = alpha * feature_reconstruction_loss + beta * structure_reconstruction_loss +\
#     gamma * proj_loss + delta * pca_loss

# print(feature_reconstruction_loss)
# print(structure_reconstruction_loss)
# print(proj_loss)
# print(pca_loss)
# print()

# return error, total_loss, feature_reconstruction_loss, structure_reconstruction_loss, proj_loss, pca_loss


'''
def FeatureOnlyVAE(feature_pred, features, alpha, beta):
    feature_reconstruction_error = torch.norm(feature_pred - features, p=2, dim=1)
    error = alpha * feature_reconstruction_error 
    feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
    total_loss = alpha * feature_reconstruction_loss
    return error, total_loss, feature_reconstruction_loss


def StructureOnlyVAE(structure_pred, adj_norm, alpha, beta):
    structure_reconstruction_error = torch.norm(structure_pred - adj_norm, p=2, dim=1)
    error =  beta * structure_reconstruction_error
    structure_reconstruction_loss = torch.mean(structure_reconstruction_error)
    total_loss =  beta * structure_reconstruction_loss
    return error, total_loss, structure_reconstruction_loss


def scat_ae_loss_function(feature_pred, structure_pred, features, adj_norm, alpha, beta):
    feature_reconstruction_error = torch.norm(feature_pred - features, p=2, dim=1)
    structure_reconstruction_error = torch.norm(structure_pred - adj_norm, p=2, dim=1)

    error = alpha * feature_reconstruction_error + beta * structure_reconstruction_error

    feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
    structure_reconstruction_loss = torch.mean(structure_reconstruction_error)

    total_loss = alpha * feature_reconstruction_loss + beta * structure_reconstruction_loss
    # total_loss = structure_reconstruction_loss
    # print(feature_reconstruction_loss)
    # print(structure_reconstruction_loss)
    # print(total_loss)
    # print()
    return error, feature_reconstruction_error, structure_reconstruction_error, total_loss, \
    feature_reconstruction_loss, structure_reconstruction_loss


def scat_ydgn_loss_function(recon_origin, recon_scat, structure_decoder_layer_2, adj_norm, alpha, beta):
    feature_reconstruction_error = torch.norm(recon_origin - recon_scat, p=2, dim=1)
    structure_reconstruction_error = torch.norm(structure_decoder_layer_2 - adj_norm, p=2, dim=1)

    error = alpha * feature_reconstruction_error + beta * structure_reconstruction_error

    feature_reconstruction_loss = torch.mean(feature_reconstruction_error)
    structure_reconstruction_loss = torch.mean(structure_reconstruction_error)

    total_loss = alpha * feature_reconstruction_loss + beta * structure_reconstruction_loss
    # total_loss = structure_reconstruction_loss
    # print(feature_reconstruction_loss)
    # print(structure_reconstruction_loss)
    # print(total_loss)
    # print()
    return error, feature_reconstruction_error, structure_reconstruction_error, total_loss, \
    feature_reconstruction_loss, structure_reconstruction_loss
'''
