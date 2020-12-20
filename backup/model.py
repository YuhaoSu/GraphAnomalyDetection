import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.encoder_gc_1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)
        
        self.feature_decoder_layer_1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)
        
        self.structure_decoder_layer_1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        #print("x.shape", x.shape)
        #print("adj.shape",adj.shape)
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        #print("encoder_layer_1.shape",encoder_layer_1.shape)
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        #print("encoder_layer_2.shape",encoder_layer_2.shape)
        return encoder_layer_2

          
    def decode(self, encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(encoder_layer_2, adj))
        #print("feature_decoder_layer_1.shape",feature_decoder_layer_1.shape)
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        #print("feature_decoder_layer_2.shape",feature_decoder_layer_2.shape)
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(self.structure_decoder_layer_1(encoder_layer_2, adj))
        #print("structure_decoder_layer_1.shape",structure_decoder_layer_1.shape)
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        #print("structure_decoder_layer_2.shape",structure_decoder_layer_2.shape)
        return feature_decoder_layer_2, structure_decoder_layer_2
    
    def forward(self, x, adj):
        encoder_layer_2 = self.encode(x, adj)
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(encoder_layer_2, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2
    
    
class GCNModelScatVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelScatVAE, self).__init__()
        '''
        self.encoder_gc_1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)
        '''
        
        self.feature_decoder_layer_1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)
        
        self.structure_decoder_layer_1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductDecoder(dropout, act=lambda x: x)

    '''
    def encode(self, x, adj):
        #print("x.shape", x.shape)
        #print("adj.shape",adj.shape)
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        #print("encoder_layer_1.shape",encoder_layer_1.shape)
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        #print("encoder_layer_2.shape",encoder_layer_2.shape)
        return encoder_layer_2
    '''
          
    def decode(self, encoder_layer_2, adj):
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(encoder_layer_2, adj))
        #print("feature_decoder_layer_1.shape",feature_decoder_layer_1.shape)
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        #print("feature_decoder_layer_2.shape",feature_decoder_layer_2.shape)
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(self.structure_decoder_layer_1(encoder_layer_2, adj))
        #print("structure_decoder_layer_1.shape",structure_decoder_layer_1.shape)
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        #print("structure_decoder_layer_2.shape",structure_decoder_layer_2.shape)
        return feature_decoder_layer_2, structure_decoder_layer_2
    
    def forward(self, y_features, adj):
        '''
        encoder_layer_2 = self.encode(x, adj)
        '''
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(y_features, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        
        return adj
   
    
class RSRlayer(nn.Module):

    def __init__(self, hidden_dim2, rsr_dim):
        super(RSRlayer, self).__init__()
        self.hidden_dim2 = hidden_dim2
        self.rsr_dim = rsr_dim
        self.rsrlayer = Parameter(torch.Tensor(self.hidden_dim2, self.rsr_dim))
        nn.init.normal_(self.rsrlayer)
        #self.rsrlayer = torch.randn(self.hidden_dim2, self.rsr_dim)
        
    def forward(self, z):
        rsr_output = torch.mm(z, self.rsrlayer)
        return self.rsrlayer, rsr_output
    
    
class GCNModelRSRAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout, rsr_dim):
        super(GCNModelRSRAE, self).__init__()
        self.encoder_gc_1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)
        
        #self.rsrlayer = Parameter(torch.Tensor(hidden_dim2, rsr_dim))
        #nn.init.normal_(self.rsrlayer)
        
        self.rsrlayer = RSRlayer(hidden_dim2, rsr_dim)
        self.rsr_normalize = nn.BatchNorm1d(rsr_dim)
        
        self.feature_decoder_layer_1 = GraphConvolution(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)
        
        self.structure_decoder_layer_1 = GraphConvolution(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        #print("x.shape", x.shape)
        #print("adj.shape",adj.shape)
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        #print("encoder_layer_1.shape",encoder_layer_1.shape)
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        #print("encoder_layer_2.shape",encoder_layer_2.shape)
        return encoder_layer_2
    
    
    def rsr(self,encoder_layer_2, rsr_dim):
        #print("encoder_layer_2",encoder_layer_2.shape)
        #encoder_flatten = torch.flatten(encoder_layer_2)
        #if device == 'gpu':
         #   rsrlayer = torch.randn(encoder_layer_2.shape[-1], rsr_dim).to(device)
        #else:
        #rsrlayer = torch.randn(encoder_layer_2.shape[-1], rsr_dim)
        
        #print("rsrlayer",rsrlayer.shape)
        #rsrlayer = self.rsrlayer
        rsr_output = torch.mm(encoder_layer_2, self.rsrlayer)
        return self.rsrlayer, rsr_output
     
    
    def decode(self, rsr_output, adj):
        #print("rsr_output",rsr_output.shape)
        rsr_output = self.rsr_normalize(rsr_output)
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(rsr_output, adj))
        #print("feature_decoder_layer_1.shape",feature_decoder_layer_1.shape)
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        #print("feature_decoder_layer_2.shape",feature_decoder_layer_2.shape)
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(self.structure_decoder_layer_1(rsr_output, adj))
        #print("structure_decoder_layer_1.shape",structure_decoder_layer_1.shape)
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        #print("structure_decoder_layer_2.shape",structure_decoder_layer_2.shape)
        return feature_decoder_layer_2, structure_decoder_layer_2
    
    def forward(self, x, adj, rsr_dim):
        encoder_layer_2 = self.encode(x, adj)
        rsrlayer, rsr_output = self.rsrlayer(encoder_layer_2)
        #rsrlayer, rsr_output = self.rsr(encoder_layer_2, rsr_dim)
        #print("rsrlayer", rsrlayer[0:5,0:5])
        #print("rsr_output", rsr_output[0:5,0:5])
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(rsr_output, adj)
        return encoder_layer_2, rsrlayer, rsr_output, feature_decoder_layer_2, structure_decoder_layer_2



class GCNModelScatRSRAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, y_feat_dim, dropout, rsr_dim):
        super(GCNModelScatRSRAE, self).__init__()
        """
        self.encoder_gc_1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)
        """
        #self.rsrlayer = Parameter(torch.Tensor(hidden_dim2, rsr_dim))
        #nn.init.normal_(self.rsrlayer)
        
        self.rsrlayer = RSRlayer(y_feat_dim, rsr_dim)
        self.rsr_normalize = nn.BatchNorm1d(rsr_dim)
        
        self.feature_decoder_layer_1 = GraphConvolution(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)
        
        self.structure_decoder_layer_1 = GraphConvolution(rsr_dim, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductDecoder(dropout, act=lambda x: x)
    """
    def encode(self, x, adj):
        #print("x.shape", x.shape)
        #print("adj.shape",adj.shape)
        encoder_layer_1 = (self.encoder_gc_1(x, adj))
        #print("encoder_layer_1.shape",encoder_layer_1.shape)
        encoder_layer_2 = self.encoder_normalize_2(self.encoder_gc_2(encoder_layer_1, adj))
        #print("encoder_layer_2.shape",encoder_layer_2.shape)
        return encoder_layer_2
    """
    
    def rsr(self,encoder_layer_2, rsr_dim):
        
        rsr_output = torch.mm(encoder_layer_2, self.rsrlayer)
        return self.rsrlayer, rsr_output
     
    
    def decode(self, rsr_output, adj):
        #print("rsr_output",rsr_output.shape)
        rsr_output = self.rsr_normalize(rsr_output)
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(rsr_output, adj))
        #print("feature_decoder_layer_1.shape",feature_decoder_layer_1.shape)
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        #print("feature_decoder_layer_2.shape",feature_decoder_layer_2.shape)
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(self.structure_decoder_layer_1(rsr_output, adj))
        #print("structure_decoder_layer_1.shape",structure_decoder_layer_1.shape)
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        #print("structure_decoder_layer_2.shape",structure_decoder_layer_2.shape)
        return feature_decoder_layer_2, structure_decoder_layer_2
    
    def forward(self, y_features, adj, rsr_dim):
        #encoder_layer_2 = self.encode(x, adj)
        
        rsrlayer, rsr_output = self.rsrlayer(y_features)
        #rsrlayer, rsr_output = self.rsr(encoder_layer_2, rsr_dim)
        #print("rsrlayer", rsrlayer[0:5,0:5])
        #print("rsr_output", rsr_output[0:5,0:5])
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(rsr_output, adj)
        return rsrlayer, rsr_output, feature_decoder_layer_2, structure_decoder_layer_2
    
    
    
class GCNModelAttScatVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAttScatVAE, self).__init__()
        '''
        self.encoder_gc_1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.encoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.encoder_gc_2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.encoder_normalize_2 = nn.BatchNorm1d(hidden_dim2)
        '''

        self.attention_1 = nn.Parameter(torch.zeros(size=(hidden_dim2, hidden_dim1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attention_2 = nn.Parameter(torch.zeros(size=(hidden_dim1, hidden_dim2)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        #self.attention_1 = torch.nn.Linear(hidden_dim2, hidden_dim1)
        #self.attention_2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        
        self.feature_decoder_layer_1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.feature_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.feature_decoder_layer_2 = GraphConvolution(hidden_dim1, input_feat_dim, dropout, act=F.relu)
        self.feature_decoder_normalize_2 = nn.BatchNorm1d(input_feat_dim)
        
        self.structure_decoder_layer_1 = GraphConvolution(hidden_dim2, hidden_dim1, dropout, act=F.relu)
        self.structure_decoder_normalize_1 = nn.BatchNorm1d(hidden_dim1)
        self.structure_decoder_layer_2 = InnerProductDecoder(dropout, act=lambda x: x)
          
    def decode(self, encoder_layer_2, adj):
        
        attention_score_base = self.attention_1(encoder_layer_2)
        attention_score_base = torch.nn.functional.relu(attention_score_base)
        attention_score = torch.nn.functional.softmax(self.attention_2(attention_score_base), dim=0)
        condensed_encoder_layer_2 = encoder_layer_2 * attention_score 
        
        feature_decoder_layer_1 = self.feature_decoder_normalize_1(self.feature_decoder_layer_1(condensed_encoder_layer_2, adj))
        #print("feature_decoder_layer_1.shape",feature_decoder_layer_1.shape)
        feature_decoder_layer_2 = self.feature_decoder_normalize_2(self.feature_decoder_layer_2(feature_decoder_layer_1, adj))
        #print("feature_decoder_layer_2.shape",feature_decoder_layer_2.shape)
        structure_decoder_layer_1 = self.structure_decoder_normalize_1(self.structure_decoder_layer_1(condensed_encoder_layer_2, adj))
        #print("structure_decoder_layer_1.shape",structure_decoder_layer_1.shape)
        structure_decoder_layer_2 = self.structure_decoder_layer_2(structure_decoder_layer_1)
        #print("structure_decoder_layer_2.shape",structure_decoder_layer_2.shape)
        return feature_decoder_layer_2, structure_decoder_layer_2
    
    def forward(self, y_features, adj):
        feature_decoder_layer_2, structure_decoder_layer_2 = self.decode(y_features, adj)
        return feature_decoder_layer_2, structure_decoder_layer_2

