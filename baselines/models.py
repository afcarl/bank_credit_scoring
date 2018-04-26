from torch import nn, functional as F
from torch.autograd import Variable
import torch

from helper import rmse, AttrProxy, BaseNet, LayerNorm


class SimpleGRU(BaseNet):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_dim, hidden_dim, output_dim, max_neighbors, neighbor_types, time_steps, dropout_prob=0.1):
        super(SimpleGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = 1
        self.time_steps = time_steps
        self.name = "SimpleRNN"
        self.rnn = nn.GRU(input_dim, hidden_dim,
                          num_layers=1,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())

        # self.prj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
        #                          nn.Tanh(),
        #                          nn.Dropout(dropout_prob),
        #                          nn.Linear(hidden_dim // 2, output_dim))



    def forward(self, input_sequence, hidden, b_neighbors_sequence, neighbor_hidden, neighbor_types, ngh_msk):
        """
        forward pass of the network
        :param input: input to the rnn
        :param hidden: hidden state
        :return:
        """
        output, hidden = self.rnn(input_sequence, hidden)
        output = self.dropout(output)
        output = self.prj(output)
        return output




class StructuralRNN(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, max_neighbors, neighbor_types, time_steps, dropout_prob=0.1):
        super(StructuralRNN, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, 1, batch_first=True, bidirectional=False)

        self.NeighborRNN = {}
        for i in range(neighbor_types):
            self.NeighborRNN[i] = nn.GRU(input_dim, hidden_dim, 1, batch_first=True, bidirectional=False)


        self.name = "StructuralRNN"
        self.out_RNN = nn.GRU(hidden_dim * (neighbor_types + 1), hidden_dim, 1, batch_first=True, bidirectional=False)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = 1
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.neighbor_types = neighbor_types
        self.criterion = nn.MSELoss()
        self.dropout = nn.Dropout(dropout_prob)




    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, neighbor_types, ngh_msk):
        batch_size = node_input.size(0)
        out_hidden = self.init_hidden(batch_size)

        out_output = Variable(torch.zeros(batch_size, self.time_steps, self.hidden_dim * self.neighbor_types))
        if torch.cuda.is_available():
            out_output.cuda()


        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        node_output = nn.functional.relu(node_output)
        node_output = self.dropout(node_output)


        neighbors_hidden = neighbors_hidden.unsqueeze(1).expand(-1, self.neighbor_types, -1, -1)
        neighbor_types = neighbor_types.unsqueeze(-1)
        neighbor_mask = torch.sum(neighbor_types, dim=1)
        neighbor_mask[neighbor_mask > 1] = 1.
        neighbors_input = neighbors_input.unsqueeze(-2) * neighbor_types
        neighbors_input = torch.sum(neighbors_input, dim=1)

        for t_idx in range(self.neighbor_types):
            output_group_by_type, hidden_group_by_type = self.NeighborRNN[t_idx](neighbors_input[:, :, t_idx], neighbors_hidden[:, t_idx])
            masked_output_group_by_type = output_group_by_type * neighbor_mask[:, :, t_idx]
            out_output[:, :, t_idx * self.hidden_dim:(t_idx + 1) * self.hidden_dim] = masked_output_group_by_type

        # # partition neighbors according to type
        # for b_idx in range(batch_size):
        #     neighbors_input_by_types = {}
        #
        #     num_neighbors = ngh_msk[b_idx] - 1
        #     n_types = torch.nonzero(neighbor_types[b_idx, :num_neighbors])[:, 1]
        #     for n_idx in range(num_neighbors):
        #         n_type = n_types[n_idx]
        #         if n_type in neighbors_input_by_types:
        #             neighbors_input_by_types[n_type] += neighbors_input[b_idx, n_idx]
        #         else:
        #             neighbors_input_by_types[n_type] = neighbors_input[b_idx, n_idx]
        #
        #     for t_idx in neighbors_input_by_types.keys():
        #         input_group_by_type = neighbors_input_by_types[t_idx]
        #         hidden_group_by_type = self.init_hidden(1)
        #         output_group_by_type, hidden_group_by_type = self.NeighborRNN[t_idx](input_group_by_type.unsqueeze(0), hidden_group_by_type)
        #         out_output[b_idx, :, t_idx * self.hidden_dim:(t_idx+1) * self.hidden_dim] = output_group_by_type[0]

        out_output = nn.functional.relu(out_output)
        out_output = self.dropout(out_output)

        output, out_hidden = self.out_RNN(torch.cat((node_output, out_output), dim=-1), out_hidden)
        output = self.dropout(output)
        output = self.prj(output)
        return output


class NodeNeighborsInterpolation(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, n_timestemps, dropout_prob=0.1):
        super(NodeNeighborsInterpolation, self).__init__()
        self.node_prj = nn.Linear(input_dim, hidden_dim)
        self.neight_prj = nn.Linear(input_dim, hidden_dim)

        self.layer_norm = LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)
        self.name = "NodeNeighborsInterpolation"

        self.prj = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                                 nn.Linear(hidden_dim, output_dim))

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = n_timestemps
        self.max_neighbors = max_neighbors

    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, s_len):

        neighbors_input = neighbors_input.sum(1)
        neighbors_output = self.neight_prj(neighbors_input)

        node_output = self.node_prj(node_input)

        output = self.prj(torch.cat((node_output, neighbors_output), dim=-1))
        return output

    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                torch.nn.init.xavier_normal(p.data)
        self.layer_norm.gamma.data.fill_(1)

class NodeInterpolation(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, n_timestemps, dropout_prob=0.1):
        super(NodeInterpolation, self).__init__()
        self.node_prj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.prj = nn.Linear(hidden_dim, output_dim)
        self.name = "NodeInterpolation"


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
        # node_output = nn.functional.relu(node_output)
        node_output = self.dropout(node_output)
        output = self.prj(node_output)
        return output



