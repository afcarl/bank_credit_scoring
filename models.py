import torch
from helper import rmse, TIMESTAMP
from numpy import convolve
import pickle
from os.path import join as path_join
from torch.autograd import Variable
import torch.nn as nn


class StructuredGuidedAttention(nn.Module):
    def __init__(self, hidden_dim, att_dim, att_hops, max_n_neig):
        super(StructuredGuidedAttention, self).__init__()
        self.att_head = nn.Sequential(nn.Linear(hidden_dim, att_dim),
                                      nn.ReLU())
        self.head_prj = nn.Linear(att_dim, att_hops)
        self.max_neighbors_number = max_n_neig
        self.hidden_dim = hidden_dim
        self.attention_dim = att_dim
        self.attention_hops = att_hops

    def forward(self, node_input, flat_neigh_input, neighbors_number):
        satcked_node_input = node_input.squeeze().repeat(self.max_neighbors_number, 1)
        sim = torch.mul(flat_neigh_input, satcked_node_input)
        attention = self.att_head(sim)/neighbors_number[0]
        attention = nn.functional.softmax(self.head_prj(attention), dim=-1)
        # TODO: add L2 regularization
        return attention.t()


class GuidedAttention(nn.Module):
    def __init__(self, hidden_dim, att_dim, max_n_neig):
        super(GuidedAttention, self).__init__()
        self.att_head = nn.Sequential(nn.Linear(hidden_dim, att_dim),
                                      nn.ReLU())
        self.max_neighbors_number = max_n_neig
        self.hidden_dim = hidden_dim
        self.attention_dim = att_dim

    def forward(self, node_input, flat_neigh_input, neighbors_number):
        satcked_node_input = node_input.squeeze().repeat(self.max_neighbors_number, 1)
        sim = torch.mul(flat_neigh_input, satcked_node_input)
        attention = self.att_head(sim)/neighbors_number[0]
        attention = nn.functional.softmax(attention, dim=-1)
        # TODO: add visualization of the weights
        return attention.t()

class SelfAttention(nn.Module):
    def __init__(self, nhid, att_dim, max_n_neig):
        super(SelfAttention, self).__init__()
        self.att_head = nn.Linear(nhid, att_dim)
        self.neigbhors_number = max_n_neig
        self.hidden_dim = nhid
        self.attention_dim = att_dim

    def forward(self, neigh_input):
        f_neigh_input = neigh_input.view(-1, neigh_input.size(-1))
        attention = self.att_head(f_neigh_input)
        attention = nn.functional.softmax(attention, dim=-1)
        return attention.t()



class SimpleStructuredNeighborAttentionRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, att_dim, att_hops, max_neighbors_n, n_timestemps, dropout_prob=0.1):
        super(SimpleStructuredNeighborAttentionRNN, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.Attention = GuidedAttention(hidden_dim, att_dim, max_neighbors_n)


        self.MLP_projection = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                            nn.ReLU())

        self.dropout = nn.Dropout(dropout_prob)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = nlayers
        self.attention_dim = att_dim
        self.criterion = nn.MSELoss()
        self.n_timestemp = n_timestemps

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
        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        f_neigh_ouput = neighbors_output.contiguous().view(-1, self.hidden_dim)
        attention = self.Attention(node_output, f_neigh_ouput, s_len)
        applied_attention = self.dropout(torch.mm(attention, f_neigh_ouput))

        output = self.MLP_projection(torch.cat((node_output.squeeze(), applied_attention), dim=0))
        return output.sum(0), node_hidden, neighbors_hidden


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

    def compute_loss(self, b_predict, b_target):
        """compute the loss"""
        return self.criterion(b_predict, b_target)

    def compute_error(self, predict, target):
        return rmse(predict, target)


