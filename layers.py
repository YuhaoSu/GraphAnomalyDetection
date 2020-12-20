
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj):
        features = F.dropout(features, self.dropout, self.training)
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductLayer(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductLayer, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, gamma, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.gamma = gamma
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.gamma)

    def forward(self, features, adj):
        #print("starting h")
        h = torch.mm(features, self.W)
        N = h.size()[0]
        '''
        print("starting h.repeat(1, N)")
        temp1 = h.repeat(1, N).view(N * N, -1)
        print("temp1 size",temp1.size())
        print("starting  h.repeat(N, 1)")
        temp2 = h.repeat(N, 1)
        # a_input = torch.cat([h.repeat(1, N).view(N * N, -1),
        #                    h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        print("temp2 size", temp2.size())
        print("starting  torch.cat")
        a_input = torch.cat([temp1, temp2],dim=1)
        print("reshaping")
        a_input.view(N, -1, 2 * self.out_features)
        '''
        h_cat = torch.cat([h,h],dim=1)#.view(1,N,2 * self.out_features)
        #print("h_cat size", h_cat.size())
        temp3 = torch.matmul(h_cat, self.a).view(1,N)
        #print("temp3 size", temp3.size())
        temp4 = torch.repeat_interleave(temp3,N,dim=0)
        #a_input = torch.repeat_interleave(h_cat,N,dim=0)
        #print("a_input size", a_input.size())
        #print("temp4 size", temp4.size())

        #print("starting leakyrelu")
        #temp3 = torch.matmul(a_input, self.a)
        e = self.leakyrelu(temp4)#.squeeze(2)
        zero_vec = -9e15*torch.ones_like(e)
        #print("starting attention")
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        #print("starting multiplication")
        h_prime = torch.matmul(attention, h)
        #print("starting output")

        if self.concat:
            output = F.elu(h_prime)
            return output
        else:
            return h_prime

    def __repr__(self):

        output = self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
        return output


class RsrLayer(nn.Module):
    def __init__(self, hidden_dim2, rsr_dim):
        super(RsrLayer, self).__init__()
        self.hidden_dim2 = hidden_dim2
        self.rsr_dim = rsr_dim
        self.rsrlayer = Parameter(torch.Tensor(self.hidden_dim2, self.rsr_dim))
        nn.init.normal_(self.rsrlayer)

    def forward(self, z):
        rsr_output = torch.mm(z, self.rsrlayer)
        return self.rsrlayer, rsr_output