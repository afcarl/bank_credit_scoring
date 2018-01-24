from torch import nn
from torch.autograd import Variable
import torch
from helper import rmse



class SimpleGRU(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, features_size, hidden_size, n_layers, output_size, batch_size, dropout=0.5):
        super(SimpleGRU, self).__init__()

        self.features_size = features_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nlayers = n_layers
        self.batch_size = batch_size

        self.rnn = nn.GRU(features_size, hidden_size[0], n_layers,
                          batch_first=True,
                          dropout=dropout)

        self.dense = nn.Sequential(nn.Linear(hidden_size[0], hidden_size[1]),
                                   nn.Linear(hidden_size[1], output_size),
                                   nn.ELU())

        self.drop = nn.Dropout(dropout)
        self.criterion = nn.MSELoss()

        self.reset_parameters()


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
        # initrange = 0.1
        # self.dense.bias.data.fill_(0)
        # self.dense.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, hidden):
        """
        forward pass of the network
        :param input: input to the rnn
        :param hidden: hidden state
        :return:
        """
        output, hidden = self.rnn(input, hidden)

        # select last output
        # row_indices = torch.arange(0, self.batch_size).long()
        # seq_length = torch.LongTensor(seq_length) - 1
        # output = output[row_indices, seq_length, :]
        output = output[:, -1, :]

        output = self.drop(output)
        output = self.dense(output)
        return output, hidden

    def init_hidden(self, batch_size):
        """
        generate a new hidden state to avoid the back-propagation to the beginning to the dataset
        :return:
        """
        weight = next(self.parameters()).data
        return Variable(weight.new(self.nlayers, batch_size, self.hidden_size[0]).zero_())

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



class SimpleConcatRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, n_timestemps, dropout_prob=0.1):
        super(SimpleConcatRNN, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        # self.MLP_projection = nn.Sequential(nn.Conv2d(1, 1, kernel_size=((max_neighbors+1) * n_timestemps, hidden_dim), stride=1),
        #                                     nn.ReLU())
        self.MLP_projection = nn.Sequential(nn.Conv2d(1, 1, kernel_size=(1, hidden_dim), stride=1),
                                             nn.ReLU())
        self.name = "RNN_Concat"
        # self.name = "RNN_concat"
        self.dropout = nn.Dropout(dropout_prob)
        self.prj = nn.Linear(hidden_dim, output_dim)

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
                nn.init.xavier_normal(p.data)

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
        output = torch.cat((node_output.unsqueeze(1), neighbors_output.view(self.batch_size, self.max_neighbors, self.n_timestemp, -1)), dim=1)
        output = self.prj(output.view(-1, self.n_timestemp, self.hidden_dim))

        # output = torch.sum(torch.sum(output.view(self.batch_size, self.max_neighbors+1, -1), dim=-1), dim=-1)
        output = torch.sum(output.view(self.batch_size, self.max_neighbors+1, -1), dim=1)
        return output, node_hidden, neighbors_hidden


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

    def compute_loss(self, predict, target):
        return self.criterion(predict, target)

    def compute_error(self, predict, target):
        return rmse(predict, target)