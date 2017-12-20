import torch
import torch.nn as nn

from torch.autograd import Variable

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
                                   nn.ReLU())

        self.drop = nn.Dropout(dropout)
        self.criterion = nn.MSELoss()

        self.reset_parameters()


    def reset_parameters(self):
        """
        reset the network parameters using xavier init
        :return:
        """
        initrange = 0.1
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