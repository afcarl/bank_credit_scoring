from torch import nn, functional as F
from torch.autograd import Variable
import torch

from helper import rmse, AttrProxy, BaseNet, get_attn_mask



class SimpleGRU(BaseNet):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, time_steps, dropout_prob=0.5):
        super(SimpleGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.name = "SimpleRNN"
        self.rnn = nn.GRU(input_dim, hidden_dim,
                          num_layers=nlayers,
                          batch_first=True)

        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())
        self.drop = nn.Dropout(dropout_prob)


    def forward(self, input_sequence, hidden, b_neighbors_sequence, neighbor_hidden,
                                                                  b_seq_len):
        """
        forward pass of the network
        :param input: input to the rnn
        :param hidden: hidden state
        :return:
        """
        output, hidden = self.rnn(input_sequence, hidden)
        output = self.drop(output)
        output = self.prj(output)
        return output




class StructuralRNN(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, n_timestemps, dropout_prob=0.1):
        super(StructuralRNN, self).__init__()
        self.NodeRNN = nn.GRU(input_dim + hidden_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)

        self.name = "StructuralRNN"

        self.dropout = nn.Dropout(dropout_prob)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.n_timestemp = n_timestemps
        self.max_neighbors = max_neighbors
        self.criterion = nn.MSELoss()


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
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        neighbors_output = neighbors_output.view(self.batch_size, self.max_neighbors, self.n_timestemp, -1)
        neighbors_output = torch.sum(neighbors_output, dim=1)

        output, node_hidden = self.NodeRNN(torch.cat((node_input, neighbors_output), dim=-1), node_hidden)
        output = self.dropout(output)
        output = self.prj(output)
        # output = torch.sum(output.view(self.batch_size, self.max_neighbors+1, -1), dim=1)
        return output

class NodeNeighborsInterpolation(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, n_timestemps, dropout_prob=0.1):
        super(NodeNeighborsInterpolation, self).__init__()
        self.node_prj = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Dropout(dropout_prob))

        self.neight_prj = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Dropout(dropout_prob))


        self.name = "NodeNeighborsInterpolation"
        self.prj = nn.Sequential(nn.Linear(2*hidden_dim, output_dim),
                                 nn.ELU())

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = n_timestemps
        self.max_neighbors = max_neighbors

    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, s_len):
        self.batch_size = node_input.size(0)

        node_output = self.node_prj(node_input)
        neighbors_output = self.neight_prj(neighbors_input.view(-1, self.time_steps, self.input_dim)).view(self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim)

        neighbors_output = torch.sum(neighbors_output, dim=1)

        output = torch.cat((node_output, neighbors_output), dim=-1)
        output = self.prj(output)
        return output



class NodeInterpolation(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, n_timestemps, dropout_prob=0.1):
        super(NodeInterpolation, self).__init__()
        self.node_prj = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Dropout(dropout_prob))

        self.name = "NodeInterpolation"
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = n_timestemps
        self.max_neighbors = max_neighbors
        self.criterion = nn.MSELoss()

    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, s_len):
        self.batch_size = node_input.size(0)

        node_output = self.node_prj(node_input)
        output = self.prj(node_output)
        return output



