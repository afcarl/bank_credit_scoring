import torch
from helper import rmse, TIMESTAMP
from numpy import convolve
import pickle
from os.path import join as path_join
from torch.autograd import Variable
import torch.nn as nn


class StructuredGuidedAttention(nn.Module):
    def __init__(self, hidden_dim, att_hops, att_dim):
        super(StructuredGuidedAttention, self).__init__()
        self.name = "StructuredGuidedAttention"
        self.S1 = nn.Linear(hidden_dim, att_dim)
        self.S2 = nn.Linear(att_dim, att_hops)

        self.hidden_dim = hidden_dim
        self.attention_dim = att_dim
        self.attention_hops = att_hops

    def forward(self, node_rnn_output, neigh_rnn_output, neighbors_number):
        self.batch_size = node_rnn_output.size(0)
        self.n_timestemp = node_rnn_output.size(1)
        self.max_neighbors_number = neigh_rnn_output.size(1)
        self.use_cuda = next(self.parameters()).is_cuda


        satcked_node_rnn_output = node_rnn_output.repeat(1, self.max_neighbors_number, 1)
        neigh_rnn_output = neigh_rnn_output.view(self.batch_size, -1, self.hidden_dim)

        if self.use_cuda:
            BA = Variable(torch.zeros(self.batch_size, self.attention_hops, self.max_neighbors_number * self.n_timestemp).cuda())
            BW = torch.zeros(self.batch_size, self.max_neighbors_number, self.n_timestemp).cuda()
            penal = Variable(torch.zeros(1).cuda())
            I = Variable(torch.eye(self.attention_hops).cuda())
        else:
            BA = Variable(torch.zeros(self.batch_size, self.attention_hops, neigh_rnn_output.size(1)))
            BW = torch.zeros(self.batch_size, self.max_neighbors_number, self.n_timestemp)
            penal = Variable(torch.zeros(1))
            I = Variable(torch.eye(self.attention_hops))

        for i in range(self.batch_size):
            H = torch.mul(neigh_rnn_output[i], satcked_node_rnn_output[i])
            s1 = self.S1(H)
            s1 = s1 / (neighbors_number[0] ** 0.5)
            s2 = self.S2(nn.functional.tanh(s1))
            # Attention Weights and Embedding
            A = nn.functional.softmax(s2.t(), dim=-1)
            BA[i] = A
            BW[i] = A.data.t().sum(-1).view(self.max_neighbors_number, -1)

            # Penalization term
            AAT = torch.mm(A, A.t())
            P = torch.norm(AAT - I, 2)
            penal += P * P

        return BW, BA, neigh_rnn_output, penal

class GuidedAttention(nn.Module):
    def __init__(self, hidden_dim, att_hops):
        super(GuidedAttention, self).__init__()
        self.name = "GuidedAttention"
        self.S1 = nn.Linear(hidden_dim, att_hops)

        self.hidden_dim = hidden_dim
        self.attention_hops = att_hops

    def forward(self, node_rnn_output, neigh_rnn_output, neighbors_number):
        self.batch_size = node_rnn_output.size(0)
        self.n_timestemp  = node_rnn_output.size(1)
        self.max_neighbors_number = neigh_rnn_output.size(1)
        self.use_cuda = next(self.parameters()).is_cuda

        satcked_node_rnn_output = node_rnn_output.repeat(1, self.max_neighbors_number, 1)
        neigh_rnn_output = neigh_rnn_output.view(self.batch_size, -1, self.hidden_dim)


        if self.use_cuda:
            BA = Variable(torch.zeros(self.batch_size, self.attention_hops, self.max_neighbors_number * self.n_timestemp).cuda())
            BW = torch.zeros(self.batch_size, self.max_neighbors_number, self.n_timestemp).cuda()
        else:
            BA = Variable(torch.zeros(self.batch_size, self.attention_hops, neigh_rnn_output.size(1)))
            BW = torch.zeros(self.batch_size, self.max_neighbors_number, self.n_timestemp)

        for i in range(self.batch_size):
            H = torch.mul(neigh_rnn_output[i], satcked_node_rnn_output[i])
            s1 = self.S1(H)
            s1 = s1.t()/(neighbors_number[0] ** 0.5)
            # Attention Weights and Embedding
            A = nn.functional.softmax(s1, dim=-1)
            BA[i] = A
            BW[i] = A.data.t().sum(-1).view(self.max_neighbors_number, -1)

            # Penalization term
            # AAT = torch.mm(A, A.t())
            # P = torch.norm(AAT - I, 2)
            # penal += P * P

        return BW, BA, neigh_rnn_output

class GuidedSelfAttention(nn.Module):
    def __init__(self, hidden_dim, max_neighbors, n_timestemp, drop_prob=0.1):
        super(GuidedSelfAttention, self).__init__()
        self.name = "GuidedSelfAttention"
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.n_timestemp = n_timestemp
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, node_rnn_output, neigh_rnn_output, neighbors_number):
        self.batch_size = node_rnn_output.size(0)

        stacked_node_rnn_output = node_rnn_output.repeat(1, self.max_neighbors, 1)
        flat_neigh_rnn_output = neigh_rnn_output.view(self.batch_size, -1, self.hidden_dim)

        S = torch.bmm(flat_neigh_rnn_output, stacked_node_rnn_output.transpose(1, 2))
        S = S.div(Variable(torch.pow(neighbors_number.float(), 0.5).unsqueeze(-1).unsqueeze(-1)))
        A = nn.functional.softmax(S, dim=-1)

        A = self.dropout(A)
        W = A.data.sum(1).view(self.batch_size, self.max_neighbors, self.n_timestemp)
        output = torch.bmm(A, flat_neigh_rnn_output)
        return output, W


class SimpleStructuredNeighborAttentionRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, n_timestemps, dropout_prob=0.1):
        super(SimpleStructuredNeighborAttentionRNN, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.Attention = GuidedSelfAttention(hidden_dim, max_neighbors, n_timestemps, dropout_prob)

        self.MLP_projection = nn.Sequential(nn.Conv2d(1, 1, kernel_size=((max_neighbors+1) * n_timestemps, hidden_dim), stride=1),
                                            nn.ReLU())
        self.name = "RNN_" + self.Attention.name

        # self.MLP_projection = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(50, hidden_dim), stride=1),
        #                                                                         nn.ReLU())
        # self.name = "RNN_concat"
        self.dropout = nn.Dropout(dropout_prob)


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = nlayers
        self.n_timestemp = n_timestemps
        self.max_neighbors = max_neighbors
        self.criterion = nn.MSELoss()

    def reset_parameters(self):
        """
        reset the network parameters using xavier init
        :return:
        """
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_uniform(p.data)

    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, s_len):
        """
        1) compute node RNN
        2) compute neighbors RNN
        3) compute attentions
        4) apply dropout on attention
        5) concatenate
        6) apply attention

        :param node_input: node input
        :param node_hidden: hidden state for node rnn
        :param neighbors_input: neighbors input
        :param neighbors_hidden: hidden state for neighbors rnn
        :param s_len: number of neighbors
        :return:
        """
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        neighbors_input = neighbors_input.view(-1, self.n_timestemp, self.input_dim)              # reduce batch dim
        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        neighbors_output = neighbors_output.contiguous().view(self.batch_size, self.max_neighbors, self.n_timestemp, self.hidden_dim) # reshape to normal dim

        applied_attention, weights = self.Attention(node_output, neighbors_output, s_len)


        output = self.MLP_projection(
            torch.cat((node_output, applied_attention), dim=1).unsqueeze(1))

        # output = self.MLP_projection(torch.cat((node_output, neighbors_output.view(self.batch_size, -1, self.hidden_dim)), dim=1).unsqueeze(1))
        output = output.squeeze()
        return output, node_hidden, neighbors_hidden, weights


    def init_hidden(self, batch_size):
        """
        generate a new hidden state to avoid the back-propagation to the beginning to the dataset
        :return:
        """
        weight = next(self.parameters()).data
        return Variable(weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

    def repackage_hidden_state(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden_state(v) for v in h)

    def compute_error(self, predict, target):
        if self.training:
            return self.criterion(predict, target)
        else:
            return rmse(predict, target)


