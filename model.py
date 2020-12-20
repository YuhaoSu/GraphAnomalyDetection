import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolutionLayer, InnerProductLayer, GraphAttentionLayer, RsrLayer


class GCNModelTwoDecodersVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelTwoDecodersVAE, self).__init__()
        self.encoder_gc_1 = GraphConvolutionLayer(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolutionLayer(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def encode(self, x, adj):
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        return encoder_layer_2

    def decode(self, encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def forward(self, x, adj):
        encoder_layer_2 = self.encode(x, adj)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(encoder_layer_2, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2

class GCNModelFeatureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelFeatureOnlyVAE, self).__init__()
        self.encoder_gc_1 = GraphConvolutionLayer(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolutionLayer(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

    def encode(self, x, adj):
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        return encoder_layer_2

    def decode(self, encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        return feature_decoder_layer_2

    def forward(self, x, adj):
        encoder_layer_2 = self.encode(x, adj)
        feature_decoder_layer_2 = self.decode(encoder_layer_2, adj)
        return feature_decoder_layer_2    
    
class GCNModelStructureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelStructureOnlyVAE, self).__init__()
        self.encoder_gc_1 = GraphConvolutionLayer(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolutionLayer(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)


        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def encode(self, x, adj):
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        return encoder_layer_2

    def decode(self, encoder_layer_2, adj):        
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return structure_decoder_layer_2

    def forward(self, x, adj):
        encoder_layer_2 = self.encode(x, adj)
        structure_decoder_layer_2 = self.decode(encoder_layer_2, adj)
        return structure_decoder_layer_2


class GCNModelRsrTwoDecodersVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, rsr_dim):
        super(GCNModelRsrTwoDecodersVAE, self).__init__()
        self.encoder_gc_1 = GraphConvolutionLayer(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolutionLayer(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)

        self.rsrlayer = RsrLayer(hidden_dim2, rsr_dim)
        self.rsr_normalize = nn.BatchNorm1d(rsr_dim)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def encode(self, x, adj):
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        return encoder_layer_2

    def rsr(self, encoder_layer_2):
        rsr_output = torch.mm(encoder_layer_2, self.rsrlayer)
        return self.rsrlayer, rsr_output

    def decode(self, rsr_output, adj):
        rsr_output = self.rsr_normalize(rsr_output)
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(rsr_output, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(rsr_output, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def forward(self, x, adj):
        encoder_layer_2 = self.encode(x, adj)
        rsrlayer, rsr_output = self.rsrlayer(encoder_layer_2)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(rsr_output, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2, encoder_layer_2, rsrlayer, rsr_output


class GCNModelRsrFeatureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, rsr_dim):
        super(GCNModelRsrFeatureOnlyVAE, self).__init__()
        self.encoder_gc_1 = GraphConvolutionLayer(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolutionLayer(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)

        self.rsrlayer = RsrLayer(hidden_dim2, rsr_dim)
        self.rsr_normalize = nn.BatchNorm1d(rsr_dim)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

    def encode(self, x, adj):
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        return encoder_layer_2

    def rsr(self, encoder_layer_2):
        rsr_output = torch.mm(encoder_layer_2, self.rsrlayer)
        return self.rsrlayer, rsr_output

    def decode(self, rsr_output, adj):
        rsr_output = self.rsr_normalize(rsr_output)
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(rsr_output, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        return feature_decoder_layer_2

    def forward(self, x, adj):
        encoder_layer_2 = self.encode(x, adj)
        rsrlayer, rsr_output = self.rsrlayer(encoder_layer_2)
        feature_decoder_layer_2 = self.decode(rsr_output, adj)
        return feature_decoder_layer_2, encoder_layer_2, rsrlayer, rsr_output


class GCNModelRsrStructureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, rsr_dim):
        super(GCNModelRsrStructureOnlyVAE, self).__init__()
        self.encoder_gc_1 = GraphConvolutionLayer(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolutionLayer(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)

        self.rsrlayer = RsrLayer(hidden_dim2, rsr_dim)
        self.rsr_normalize = nn.BatchNorm1d(rsr_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def encode(self, x, adj):
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        return encoder_layer_2

    def rsr(self, encoder_layer_2):
        rsr_output = torch.mm(encoder_layer_2, self.rsrlayer)
        return self.rsrlayer, rsr_output

    def decode(self, rsr_output, adj):
        rsr_output = self.rsr_normalize(rsr_output)
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(rsr_output, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return structure_decoder_layer_2

    def forward(self, x, adj):
        encoder_layer_2 = self.encode(x, adj)
        rsrlayer, rsr_output = self.rsrlayer(encoder_layer_2)
        structure_decoder_layer_2 = self.decode(rsr_output, adj)
        return structure_decoder_layer_2, encoder_layer_2, rsrlayer, rsr_output


class GCNModelRsrScatTwoDecodersVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, rsr_dim):
        super(GCNModelRsrScatTwoDecodersVAE, self).__init__()
        self.rsrlayer = RsrLayer(hidden_dim2, rsr_dim)
        self.rsr_normalize = nn.BatchNorm1d(rsr_dim)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def rsr(self, encoder_layer_2):
        rsr_output = torch.mm(encoder_layer_2, self.rsrlayer)
        return self.rsrlayer, rsr_output

    def decode(self, rsr_output, adj):
        rsr_output = self.rsr_normalize(rsr_output)
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(rsr_output, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(rsr_output, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        rsrlayer, rsr_output = self.rsrlayer(encoder_layer_2)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(rsr_output, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2, encoder_layer_2, rsrlayer, rsr_output


class GCNModelRsrScatFeatureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, rsr_dim):
        super(GCNModelRsrScatFeatureOnlyVAE, self).__init__()
        self.rsrlayer = RsrLayer(hidden_dim2, rsr_dim)
        self.rsr_normalize = nn.BatchNorm1d(rsr_dim)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

    def rsr(self, encoder_layer_2):
        rsr_output = torch.mm(encoder_layer_2, self.rsrlayer)
        return self.rsrlayer, rsr_output

    def decode(self, rsr_output, adj):
        rsr_output = self.rsr_normalize(rsr_output)
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(rsr_output, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        return feature_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        rsrlayer, rsr_output = self.rsrlayer(encoder_layer_2)
        feature_decoder_layer_2 = self.decode(rsr_output, adj)
        return feature_decoder_layer_2, rsrlayer, rsr_output


class GCNModelRsrScatStructureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, rsr_dim):
        super(GCNModelRsrScatStructureOnlyVAE, self).__init__()
        self.rsrlayer = RsrLayer(hidden_dim2, rsr_dim)
        self.rsr_normalize = nn.BatchNorm1d(rsr_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def rsr(self, encoder_layer_2):
        rsr_output = torch.mm(encoder_layer_2, self.rsrlayer)
        return self.rsrlayer, rsr_output

    def decode(self, rsr_output, adj):
        rsr_output = self.rsr_normalize(rsr_output)
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(rsr_output, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        rsrlayer, rsr_output = self.rsrlayer(encoder_layer_2)
        structure_decoder_layer_2 = self.decode(rsr_output, adj)
        return structure_decoder_layer_2, rsrlayer, rsr_output


class GCNModelScatTwoDecodersVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelScatTwoDecodersVAE, self).__init__()
        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def decode(self, encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def forward(self, y_features, adj):
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(y_features, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2    
    
class GCNModelScatFeatureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelScatFeatureOnlyVAE, self).__init__()
        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

    def decode(self, encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        return feature_decoder_layer_2

    def forward(self, y_features, adj):
        feature_decoder_layer_2 = self.decode(y_features, adj)
        return feature_decoder_layer_2
    
class GCNModelScatStructureOnlyVAE(nn.Module):
    
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelScatStructureOnlyVAE, self).__init__()
        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def decode(self, encoder_layer_2, adj):
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return structure_decoder_layer_2

    def forward(self, y_features, adj):
        structure_decoder_layer_2 = self.decode(y_features, adj)
        return structure_decoder_layer_2


class GCNModelStrAttScatTwoDecodersVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelStrAttScatTwoDecodersVAE, self).__init__()

        self.dropout = dropout
        self.s_attention_1 = GraphAttentionLayer(hidden_dim2 , hidden_dim2, self.dropout, gamma=0.2, concat=False)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def structure_attention(self,condensed_encoder_layer_2,adj):
        condensed_encoder_layer_2 = self.s_attention_1(condensed_encoder_layer_2, adj)
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(condensed_encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(condensed_encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.structure_attention(encoder_layer_2, adj)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2
        
class GCNModelStrAttScatFeatureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelStrAttScatFeatureOnlyVAE, self).__init__()

        self.dropout = dropout
        self.s_attention_1 = GraphAttentionLayer(hidden_dim2 , hidden_dim2, self.dropout, gamma=0.2, concat=False)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)


    def structure_attention(self,condensed_encoder_layer_2,adj):
        condensed_encoder_layer_2 = self.s_attention_1(condensed_encoder_layer_2, adj)
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(condensed_encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        return feature_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.structure_attention(encoder_layer_2, adj)
        feature_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return feature_decoder_layer_2
       
class GCNModelStrAttScatStructureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelStrAttScatStructureOnlyVAE, self).__init__()

        self.dropout = dropout
        self.s_attention_1 = GraphAttentionLayer(hidden_dim2 , hidden_dim2, self.dropout, gamma=0.2, concat=False)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def structure_attention(self,condensed_encoder_layer_2,adj):
        condensed_encoder_layer_2 = self.s_attention_1(condensed_encoder_layer_2, adj)
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
        self.structure_decoder_layer_1(condensed_encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.structure_attention(encoder_layer_2, adj)
        structure_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return structure_decoder_layer_2
    

class GCNModelDoubleAttScatTwoDecodersVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelDoubleAttScatTwoDecodersVAE, self).__init__()

        self.dropout = dropout
        self.attention_1 = torch.nn.Linear(hidden_dim2, hidden_dim1, bias=False)
        self.attention_2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        self.s_attention_1 = GraphAttentionLayer(hidden_dim2 , hidden_dim2, self.dropout, gamma=0.2, concat=False)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)
        
    def feature_attention(self, encoder_layer_2):
    
        attention_score_base = self.attention_1(encoder_layer_2)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = torch.nn.functional.softmax(self.attention_2(attention_score_base), dim=1)
        condensed_encoder_layer_2 = encoder_layer_2 * attention_score
        return condensed_encoder_layer_2

    def structure_attention(self,condensed_encoder_layer_2,adj):
        condensed_encoder_layer_2 = self.s_attention_1(condensed_encoder_layer_2, adj)
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(condensed_encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(condensed_encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.feature_attention(encoder_layer_2)
        condensed_encoder_layer_2 = self.structure_attention(condensed_encoder_layer_2, adj)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2

class GCNModelDoubleAttScatFeatureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelDoubleAttScatFeatureOnlyVAE, self).__init__()

        self.dropout = dropout
        self.attention_1 = torch.nn.Linear(hidden_dim2, hidden_dim1, bias=False)
        self.attention_2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        self.s_attention_1 = GraphAttentionLayer(hidden_dim2 , hidden_dim2, self.dropout, gamma=0.2, concat=False)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)
        
    def feature_attention(self, encoder_layer_2):
        attention_score_base = self.attention_1(encoder_layer_2)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = torch.nn.functional.softmax(self.attention_2(attention_score_base), dim=1)
        condensed_encoder_layer_2 = encoder_layer_2 * attention_score
        return condensed_encoder_layer_2

    def structure_attention(self,condensed_encoder_layer_2,adj):
        condensed_encoder_layer_2 = self.s_attention_1(condensed_encoder_layer_2, adj)
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(condensed_encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        return feature_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.feature_attention(encoder_layer_2)
        condensed_encoder_layer_2 = self.structure_attention(condensed_encoder_layer_2, adj)
        feature_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return feature_decoder_layer_2

class GCNModelDoubleAttScatStructureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelDoubleAttScatStructureOnlyVAE, self).__init__()

        self.dropout = dropout
        self.attention_1 = torch.nn.Linear(hidden_dim2, hidden_dim1, bias=False)
        self.attention_2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False)
        self.s_attention_1 = GraphAttentionLayer(hidden_dim2 , hidden_dim2, self.dropout, gamma=0.2, concat=False)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)
        
    def feature_attention(self, encoder_layer_2):
        attention_score_base = self.attention_1(encoder_layer_2)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = torch.nn.functional.softmax(self.attention_2(attention_score_base), dim=1)
        condensed_encoder_layer_2 = encoder_layer_2 * attention_score
        return condensed_encoder_layer_2

    def structure_attention(self,condensed_encoder_layer_2,adj):
        condensed_encoder_layer_2 = self.s_attention_1(condensed_encoder_layer_2, adj)
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(condensed_encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return  structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.feature_attention(encoder_layer_2)
        condensed_encoder_layer_2 = self.structure_attention(condensed_encoder_layer_2, adj)
        structure_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return structure_decoder_layer_2



class GCNModelFeaAttScatTwoDecodersVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelFeaAttScatTwoDecodersVAE, self).__init__()

        self.dropout = dropout
        self.attention_1 = torch.nn.Linear(hidden_dim2, hidden_dim1, bias=False)
        self.attention_2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)
        
    def feature_attention(self, encoder_layer_2):
    
        attention_score_base = self.attention_1(encoder_layer_2)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = torch.nn.functional.softmax(self.attention_2(attention_score_base), dim=1)
        condensed_encoder_layer_2 = encoder_layer_2 * attention_score
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(condensed_encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(condensed_encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.feature_attention(encoder_layer_2)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2

class GCNModelFeaAttScatFeatureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelFeaAttScatFeatureOnlyVAE, self).__init__()

        self.dropout = dropout
        self.attention_1 = torch.nn.Linear(hidden_dim2, hidden_dim1, bias=False)
        self.attention_2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)
        
    def feature_attention(self, encoder_layer_2):
    
        attention_score_base = self.attention_1(encoder_layer_2)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = torch.nn.functional.softmax(self.attention_2(attention_score_base), dim=1)
        condensed_encoder_layer_2 = encoder_layer_2 * attention_score
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(condensed_encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        return feature_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.feature_attention(encoder_layer_2)
        feature_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return feature_decoder_layer_2

class GCNModelFeaAttScatStructureOnlyVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelFeaAttScatStructureOnlyVAE, self).__init__()

        self.dropout = dropout
        self.attention_1 = torch.nn.Linear(hidden_dim2, hidden_dim1, bias=False)
        self.attention_2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)
        
    def feature_attention(self, encoder_layer_2):
    
        attention_score_base = self.attention_1(encoder_layer_2)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = torch.nn.functional.softmax(self.attention_2(attention_score_base), dim=1)
        condensed_encoder_layer_2 = encoder_layer_2 * attention_score
        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(condensed_encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return  structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.feature_attention(encoder_layer_2)
        structure_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return structure_decoder_layer_2


class GCNModelAttScatVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAttScatVAE, self).__init__()
        """
        # Another way to express feature based attention
        self.attention_1 = nn.Parameter(torch.zeros(size=(hidden_dim2, hidden_dim1)))
        nn.init.xavier_uniform_(self.attention_1.data, gain=1.414)
        self.attention_2 = nn.Parameter(torch.zeros(size=(hidden_dim1, hidden_dim2)))
        nn.init.xavier_uniform_(self.attention_2.data, gain=1.414)
        """

        self.attention_1 = torch.nn.Linear(hidden_dim2, hidden_dim1, bias=False)
        self.attention_2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=False)

        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)

    def feature_attention(self, encoder_layer_2):
        """
        # Another way to express feature based attention
        h = torch.mm(encoder_layer_2, self.attention_1)
        h = torch.nn.functional.relu(h)
        h = torch.mm(h, self.attention_2)
        h = torch.nn.functional.softmax(h, dim=1)
        condensed_encoder_layer_2 = encoder_layer_2 * h
        """
        attention_score_base = self.attention_1(encoder_layer_2)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = torch.nn.functional.softmax(self.attention_2(attention_score_base), dim=1)
        #np.savetxt("attention_score.csv", attention_score.detach().numpy(), delimiter=',')
        condensed_encoder_layer_2 = encoder_layer_2 * attention_score
        #np.savetxt("encoder_layer_2.csv", encoder_layer_2.detach().numpy(), delimiter=',')
        #np.savetxt("condensed_encoder_layer_2.csv", condensed_encoder_layer_2.detach().numpy(), delimiter=',')

        return condensed_encoder_layer_2

    def decode(self, condensed_encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
            self.feature_decoder_layer_1(condensed_encoder_layer_2, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
            self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
            self.structure_decoder_layer_1(condensed_encoder_layer_2, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def forward(self, encoder_layer_2, adj):
        condensed_encoder_layer_2 = self.feature_attention(encoder_layer_2)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(condensed_encoder_layer_2, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2

class GCNModelDoubleAttScatYDGN(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelDoubleAttScatYDGN, self).__init__()
        self.dropout = dropout

        #attention for features after scatting
        self.s_attentions_scat = [GraphAttentionLayer(input_feat_dim, input_feat_dim, self.dropout, gamma=0.2, concat=True) for _ in
                           range(10)] #nheads=10
        for i, attention in enumerate(self.s_attentions_scat):
            self.add_module('attention_{}'.format(i), attention)
        self.s_out_att_scat = GraphAttentionLayer(input_feat_dim * 10, input_feat_dim, self.dropout, gamma=0.2, concat=False)

        #attention for features
        self.s_attentions_origin = [GraphAttentionLayer(input_feat_dim, input_feat_dim, self.dropout, gamma=0.2, concat=True) for _ in
                           range(10)] #nheads=10
        for i, attention in enumerate(self.s_attentions_origin):
            self.add_module('attention_{}'.format(i), attention)
        self.s_out_att_origin = GraphAttentionLayer(input_feat_dim * 10, input_feat_dim, self.dropout, gamma=0.2, concat=False)

        #decoder for features after scatting
        self.feature_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolutionLayer(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)

        self.structure_decoder_layer_1 = GraphConvolutionLayer(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductLayer(dropout, act=lambda x: x)


    def decode_scat(self, condensed_encoder_scat, adj):

        feature_decoder_layer_1 = self.feature_decoder_normalize_1(
        self.feature_decoder_layer_1(condensed_encoder_scat, adj))
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(
        self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(
        self.structure_decoder_layer_1(condensed_encoder_scat, adj))
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        return feature_decoder_layer_2, structure_decoder_layer_2

    def att_scat(self, feature_decoder_layer_2, adj):
        condensed_encoder_layer_2 = F.dropout(feature_decoder_layer_2, self.dropout, training=self.training)
        condensed_encoder_layer_2 = torch.cat([att(condensed_encoder_layer_2, adj) for att in self.s_attentions_scat], dim=1)
        condensed_encoder_layer_2 = F.dropout(condensed_encoder_layer_2, self.dropout, training=self.training)
        condensed_encoder_scat = self.s_out_att_scat(condensed_encoder_layer_2, adj)
        return condensed_encoder_scat

    def att_origin(self,origin,adj):
        condensed_encoder_layer_2 = F.dropout(origin, self.dropout, training=self.training)
        condensed_encoder_layer_2 = torch.cat([att(condensed_encoder_layer_2, adj) for att in self.s_attentions_origin], dim=1)
        condensed_encoder_layer_2 = F.dropout(condensed_encoder_layer_2, self.dropout, training=self.training)
        condensed_encoder_origin = self.s_out_att_origin(condensed_encoder_layer_2, adj)
        return condensed_encoder_origin

    def forward(self, condensed_encoder_scat, origin, adj):
        recon_origin = self.att_origin(origin,adj)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode_scat(condensed_encoder_scat, adj)
        recon_scat = self.att_scat(feature_decoder_layer_2,adj)

        return recon_origin, recon_scat, structure_decoder_layer_2