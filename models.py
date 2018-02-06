import torch
from helper import rmse, TIMESTAMP, get_attn_mask, get_temperature, BaseNet
from torch.autograd import Variable
import torch.nn as nn
from attention_moduls import FeatureJointAttention, TimeAttention, JointAttention, BiLinearProjection, ClusteringExamples, NetAttention, LayerNormalization




class TestNetAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, time_steps, n_heads, dropout_prob=0.1):
        super(TestNetAttention, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NetAttention = NetAttention(n_heads, hidden_dim, max_neighbors, time_steps, dropout_prob)

        self.name = "RNN" + self.NetAttention.name
        self.dropout = nn.Dropout(dropout_prob)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.n_heads = n_heads

        self.reset_parameters()

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

        neighbors_input = neighbors_input.view(-1, self.time_steps, self.input_dim)  # reduce batch dim
        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        neighbors_output = neighbors_output.contiguous().view(self.batch_size, self.max_neighbors, self.time_steps,
                                                              self.hidden_dim)  # reshape to normal dim

        app_attention, weights = self.NetAttention(node_output, neighbors_output, s_len)

        output = torch.sum(torch.cat((node_output.unsqueeze(1), app_attention), dim=1), dim=1)
        output = output.squeeze()
        return output, node_hidden, neighbors_hidden, weights.data.cpu()

class TestTimeAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, time_steps, n_heads, dropout_prob=0.1):
        super(TestTimeAttention, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.TimeAttention = TimeAttention(n_heads, hidden_dim, max_neighbors, time_steps, dropout_prob)
        self.prj = nn.Linear(hidden_dim, output_dim)
        self.name = "RNN" + self.TimeAttention.name
        self.dropout = nn.Dropout(dropout_prob)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.n_heads = n_heads

        self.reset_parameters()


    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, s_len):
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        neighbors_input = neighbors_input.view(-1, self.time_steps, self.input_dim)              # reduce batch dim
        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        neighbors_output = neighbors_output.contiguous().view(self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim) # reshape to normal dim
        attn_mask = get_attn_mask(node_input.size()[:-1], self.use_cuda)

        time_attention, time_weights = self.TimeAttention(node_output, neighbors_output, s_len, attn_mask=attn_mask)

        output = self.prj(time_attention.view(-1, self.time_steps, self.hidden_dim))
        output = torch.sum(output.view(self.batch_size, self.max_neighbors+1, -1), dim=1)
        # output = torch.sum(output.view(self.batch_size, self.max_neighbors, -1), dim=1)

        output = output.squeeze()
        return output, node_hidden, neighbors_hidden, time_weights.cpu()


class JointSelfAttentionRNN(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, time_steps, time_window, dropout_prob=0.1):
        super(JointSelfAttentionRNN, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.Attention = JointAttention(hidden_dim, max_neighbors, time_steps, dropout_prob)
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


    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, s_len):
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
        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        neighbors_output = neighbors_output.contiguous().view(self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim) # reshape to normal dim
        
        atten_mask = get_attn_mask(neighbors_output.size(), self.use_cuda, self.time_window)
        output, attention = self.Attention(node_output, neighbors_output, s_len, atten_mask)
        output = self.dropout(output)
        output = self.prj(output)

        output = output.squeeze()
        return output, node_hidden, neighbors_hidden, attention


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
        attentions = torch.FloatTensor(self.batch_size, self.time_steps, self.max_neighbors+1, self.time_steps).zero_()
        att_norms = []
        for i in range(self.time_steps):
            node_hidden = self.NodeRNN(node_input[:, i], node_hidden)
            neighbors_hidden = self.NeighborRNN(neighbors_input[:, i], neighbors_hidden)
            node_output[:, i] = node_hidden
            neighbors_output[:, :, i] = neighbors_hidden.contiguous().view(self.batch_size, self.max_neighbors, self.hidden_dim) # reshape to normal dim
            atten_mask = get_attn_mask(neighbors_output.size(), self.use_cuda, self.time_window)


            # attend to current row
            if i < self.time_steps - 1:
                atten_mask[:, :, i+1:] = 1
                atten_mask[:, i+1:] = 1
                if i > 0:
                    atten_mask[:, :i] = 1


            output, attention, att_norm = self.Attention(node_output, neighbors_output, contitional_outputs, s_len, i, atten_mask)
            att_norms.append(att_norm)

            output = self.dropout(output)
            output = self.prj(output)
            outputs[:, i] = output
            attentions[:, i] = attention

            upper_bound = i+1
            lower_bound = i-2 if i >= 3 else 0
            diff = (outputs[:, lower_bound:upper_bound, 0] - target[:, lower_bound:upper_bound, 0])
            if i < 2:
                contitional_outputs[:, i+1] = torch.cat((diff, Variable(torch.FloatTensor(self.batch_size, 3-(i+1)).zero_())), dim=-1)
            elif i < self.time_steps-1:
                contitional_outputs[:, i+1] = diff

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


