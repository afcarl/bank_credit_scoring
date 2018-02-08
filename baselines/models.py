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

        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim))
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
        return output, hidden, neighbor_hidden




class StructuralRNN(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, n_timestemps, dropout_prob=0.1):
        super(StructuralRNN, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)

        self.name = "StructuralRNN"

        self.dropout = nn.Dropout(dropout_prob)
        self.prj = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim),
                                 nn.Linear(hidden_dim, output_dim),
                                 nn.ReLU())

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
        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        neighbors_output = neighbors_output.view(self.batch_size, self.max_neighbors, self.n_timestemp, -1)

        output = torch.cat((node_output, torch.sum(neighbors_output, dim=1)), dim=-1)
        output = self.dropout(output)
        output = self.prj(output)
        # output = torch.sum(output.view(self.batch_size, self.max_neighbors+1, -1), dim=1)
        return output, node_hidden, neighbors_hidden

class FeaturedJointSelfAttentionRNN(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, time_steps, time_window, dropout_prob=0.1):
        super(FeaturedJointSelfAttentionRNN, self).__init__()
        self.NodeRNN = nn.GRUCell(input_dim, hidden_dim)
        self.NeighborRNN = nn.GRUCell(input_dim, hidden_dim)
        self.Attention = FeatureJointAttention(hidden_dim, max_neighbors, time_steps)
        self.prj = nn.Linear(hidden_dim, output_dim)
        self.name = "RNN" + self.Attention.name
        self.dropout = nn.Dropout(dropout_prob)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.time_window = time_window
        self.criterion = nn.MSELoss()

    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, s_len, target):
        """
        1) compute node RNN
        2) compute neighbors RNN
        3) compute attentions
        4) apply dropout on attention

        :param node_input: node input
        :param node_hidden: hidden state for node rnn
        :param neighbors_input: neighbors input
        :param neighbors_hidden: hidden state for neighbors rnn
        :param s_len: number of neighbors
        :return:
        """
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        neighbors_input = neighbors_input.view(-1, self.time_steps, self.input_dim)
        node_output = Variable(torch.FloatTensor(self.batch_size, self.time_steps, self.hidden_dim).zero_())
        neighbors_output = Variable(torch.FloatTensor(self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim).zero_())
        outputs = Variable(torch.FloatTensor(self.batch_size, self.time_steps, 1).zero_())
        contitional_outputs = Variable(torch.FloatTensor(self.batch_size, self.time_steps, 3).zero_())
        attentions = torch.FloatTensor(self.batch_size, self.time_steps, self.max_neighbors + 1, self.time_steps).zero_()
        if self.use_cuda:
            node_output = node_output.cuda()
            neighbors_output = neighbors_output.cuda()
            outputs = outputs.cuda()
            contitional_outputs = contitional_outputs.cuda()
            attentions = attentions.cuda()

        att_norms = []
        for i in range(self.time_steps):
            node_hidden = self.NodeRNN(node_input[:, i], node_hidden)
            neighbors_hidden = self.NeighborRNN(neighbors_input[:, i], neighbors_hidden)
            node_output[:, i] = node_hidden
            neighbors_output[:, :, i] = neighbors_hidden.contiguous().view(self.batch_size, self.max_neighbors,
                                                                           self.hidden_dim)  # reshape to normal dim
            atten_mask = get_attn_mask(neighbors_output.size(), self.use_cuda, self.time_window)

            # attend to current row
            if i < self.time_steps - 1:
                atten_mask[:, :, i + 1:] = 1
                atten_mask[:, i + 1:] = 1
                if i > 0:
                    atten_mask[:, :i] = 1

            output, attention, att_norm = self.Attention(node_output, neighbors_output, contitional_outputs, s_len, i, atten_mask)
            att_norms.append(att_norm)

            output = self.dropout(output)
            output = self.prj(output)
            outputs[:, i] = output
            attentions[:, i] = attention

            upper_bound = i + 1
            lower_bound = i - 2 if i >= 3 else 0
            diff = (outputs[:, lower_bound:upper_bound, 0] - target[:, lower_bound:upper_bound, 0])
            if i < 2:
                if self.use_cuda:
                    padd = Variable(torch.cuda.FloatTensor(self.batch_size, 3 - (i + 1)).zero_())
                else:
                    padd = Variable(torch.FloatTensor(self.batch_size, 3 - (i + 1)).zero_())
                contitional_outputs[:, i + 1] = torch.cat((diff, padd), dim=-1)
            elif i < self.time_steps - 1:
                contitional_outputs[:, i + 1] = diff

            # contitional_outputs[:, i, 0:min(upper_bound, 3)] =

        # outputs = self.dropout(outputs)
        # outputs = self.prj(outputs)
        return outputs, node_hidden, neighbors_hidden, attentions, torch.mean(torch.mean(torch.stack(att_norms, dim=0), dim=0))

    def init_hidden(self, batch_size):
        """
        generate a new hidden state to avoid the back-propagation to the beginning to the dataset
        :return:
        """
        weight = next(self.parameters()).data
        return Variable(weight.new(batch_size, self.hidden_dim).zero_())


class FeatureJointAttention(nn.Module):
    def __init__(self, hidden_dim, max_neighbors, time_steps):
        super(FeatureJointAttention, self).__init__()
        self.name = "_FeatureJointAttention"

        self.softmax = nn.Softmax(dim=-1)

        self.proj_query = nn.Linear(hidden_dim + 3, hidden_dim, bias=False)
        self.proj_key = nn.Linear(hidden_dim + 3, hidden_dim, bias=False)
        self.proj_value = nn.Linear(hidden_dim + 3, hidden_dim, bias=False)
        # self.proj_node_output = nn.Linear(hidden_dim + 3, hidden_dim)

        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.temp = ((max_neighbors+1) * time_steps) ** 0.5
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_normal(p.data)

    def forward(self, node_rnn_output, neigh_rnn_output, conditional_output, neighbors_number, input_time_steps, attn_mask=None):
        self.batch_size = node_rnn_output.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        node_rnn_output = torch.cat((node_rnn_output, conditional_output), dim=-1)
        neigh_rnn_output = torch.cat((neigh_rnn_output, conditional_output.unsqueeze(1).expand(-1, self.max_neighbors, -1, -1)), dim=-1)

        querys = node_rnn_output.unsqueeze(1).expand(-1, self.max_neighbors+1, -1, -1)
        querys = querys.contiguous().view(-1, self.time_steps, self.hidden_dim + 3)
        key_values = torch.cat((node_rnn_output.unsqueeze(1), neigh_rnn_output), dim=1)

        w_querys = self.proj_query(querys)
        w_key = self.proj_key(key_values.view(-1, self.time_steps, self.hidden_dim + 3))
        w_values = self.proj_value(key_values.view(self.batch_size, -1, self.hidden_dim + 3))

        S = torch.bmm(w_querys, w_key.transpose(1, 2))
        S = S.view(self.batch_size, self.max_neighbors+1, self.time_steps, self.time_steps)
        S = S[:, :, input_time_steps]
        S_norm = torch.norm(S.data.contiguous().view(self.batch_size, -1), 2, -1)

        if attn_mask is not None:
            S.data.masked_fill_(attn_mask[:, input_time_steps].unsqueeze(1).expand(-1, self.max_neighbors+1, -1), -float('inf'))
        S = S.contiguous().view(self.batch_size, 1, -1)
        # self.proj_cond_output(conditional_output)

        # s = [self.proj(querys[:, i], key_values[:, i]) for i in range(self.max_neighbors+1)]
        # s = [m.data.masked_fill_(attn_mask, -float('inf')) for m in s]
        # s_cat = torch.stack(s, dim=2)
        #
        # assert (S.data == s_cat).any()

        A = self.softmax(S.contiguous().view(self.batch_size, 1, -1))
        # output = self.proj_v(A, values)
        output = torch.bmm(A, w_values).squeeze()
        # output = torch.bmm(A, key_values.view(self.batch_size, -1, self.hidden_dim))
        return output, A.data.view(self.batch_size, self.max_neighbors+1, -1), S_norm

