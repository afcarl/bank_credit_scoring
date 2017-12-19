import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torch.autograd import Variable
import torch.optim as optim


class Net(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_size, hidden_size, n_layers, output_size, batch_size, dropout=0.5):
        super(Net, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.nlayers = n_layers
        self.batch_size = batch_size

        self.rnn = nn.GRU(input_size, hidden_size, n_layers,
                          batch_first=True,
                          dropout=dropout)


        self.dense = nn.Sequential(nn.Linear(hidden_size, output_size),
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
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform(p)


    def forward(self, input, hidden):
        """
        forward pass of the network
        :param input: input to the rnn
        :param hidden: hidden state
        :return:
        """
        output, hidden = self.rnn(input, hidden)

        # select last output (wrong in this case)
        # row_indices = torch.arange(0, self.batch_size).long()
        # seq_length = torch.LongTensor(seq_length) - 1
        # output = output[row_indices, seq_length, :]

        output = self.drop(output)
        output = self.dense(output)
        output = self.drop(output)
        return output

    def init_hidden(self):
        """
        generate a new hidden state to avoid the back-propagation to the beginning to the dataset
        :return:
        """
        return Variable(torch.zeros(self.nlayers, self.batch_size, self.hidden_size))

    def compute_loss(self, b_predict, b_target):
        """compute the loss"""
        return self.criterion(b_predict, b_target)