import torch
from helper import rmse, TIMESTAMP, get_attn_mask, get_temperature, BaseNet, TENSOR_TYPE, hookFunc, PositionwiseFeedForward
from torch.autograd import Variable
import torch.nn as nn
from attention_moduls import TimeAttention, FeatureTransformerLayer, TransformerLayer
import numpy as np




class TestNetAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, n_head, nlayers, max_neighbors, time_steps, time_window, temperature=1, dropout_prob=0.1):
        super(TestNetAttention, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)

        self.attention = TransformerLayer(n_head, hidden_dim, hidden_dim, max_neighbors, time_steps, temperature, dropout_prob)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        self.name = "TestNet" + self.attention.name
        self.dropout = nn.Dropout(dropout_prob)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors

    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                torch.nn.init.xavier_uniform(p.data)
        self.attention.layer_norm.gamma.data.fill_(1)

    def forward(self, node_input, neighbors_input, ngh_msk, node_hidden, neighbors_hidden, target):
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
        :param ngh_msk: number of neighbors
        :return:
        """
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        neighbors_input = neighbors_input.view(-1, self.time_steps, self.input_dim)  # reduce batch dim
        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        neighbors_output = neighbors_output.contiguous().view(self.batch_size, self.max_neighbors, self.time_steps,
                                                              self.hidden_dim)  # reshape to normal dim
        attn_mask = get_attn_mask(ngh_msk, 1, neighbors_output.size(), self.use_cuda)
        # node_output = nn.functional.relu(node_output)
        # neighbors_output = nn.functional.relu(neighbors_output)


        output, weights = self.attention(node_output, neighbors_output, attn_mask)
        output = self.dropout(output)
        return self.proj(output), weights

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


class TranslatorJointAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, n_head, nlayers, max_neighbors, time_steps, time_window, dropout_prob=0.1, temperature=1):
        super(TranslatorJointAttention, self).__init__()
        self.node_enc = nn.Linear(input_dim, hidden_dim)
        self.neigh_enc = nn.Linear(input_dim, hidden_dim)

        self.attention = TransformerLayer(n_head, hidden_dim, hidden_dim, max_neighbors, time_steps, temperature=temperature, dropout=dropout_prob)
        self.poss_wise = PositionwiseFeedForward(hidden_dim, 2*hidden_dim, dropout_prob)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, output_dim))

        self.name = "Translator" + self.attention.name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.time_window = time_window
        self.temperature = temperature
        self.n_ead = n_head




    def forward(self, node_input, neighbors_input, ngh_msk, node_hidden, neighbors_hidden, target):
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        attn_mask = get_attn_mask(ngh_msk, self.time_window, neighbors_input.size(), self.use_cuda)
        node_input = self.node_enc(node_input)
        neighbors_input = self.neigh_enc(neighbors_input)

        output, slf_att = self.attention(node_input, neighbors_input, attn_mask)
        output = self.poss_wise(output)
        output = self.proj(output).squeeze()
        return output, slf_att

    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                torch.nn.init.xavier_normal(p.data)
        self.attention.layer_norm.gamma.data.fill_(1)
        # self.attention.pos_ffn.layer_norm.a_2.data.fill_(1)



class RNNJointAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, n_head, nlayers, max_neighbors, time_steps, time_window, dropout_prob=0.1, temperature=1):
        super(RNNJointAttention, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.NeighborRNN = nn.GRU(input_dim, hidden_dim, nlayers, batch_first=True, bidirectional=False)
        self.attention = TransformerLayer(n_head, hidden_dim, hidden_dim, max_neighbors, time_steps, temperature=temperature, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        # self.attention = SingleHeadAttention(hidden_dim, max_neighbors, time_steps, temperature=temperature, dropout=dropout_prob)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim))

                    # nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                    #              nn.Tanh(),
                    #              nn.Dropout(dropout_prob),
                    #              nn.Linear(hidden_dim // 2, output_dim))

        self.name = "RNN" + self.attention.name

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.time_window = time_window
        self.temperature = temperature
        self.n_head = n_head

    def forward(self, node_input, neighbors_input, ngh_msk, node_hidden, neighbors_hidden, target):
        """
        1) compute node RNN
        2) compute neighbors RNN
        3) compute attentions
        4) apply dropout on attention

        :param node_input: node input
        :param node_hidden: hidden state for node rnn
        :param neighbors_input: neighbors input
        :param neighbors_hidden: hidden state for neighbors rnn
        :param ngh_msk: number of neighbors
        :return:
        """
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        neighbors_input = neighbors_input.view(-1, self.time_steps, self.input_dim)
        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        neighbors_output, neighbors_hidden = self.NeighborRNN(neighbors_input, neighbors_hidden)
        neighbors_output = neighbors_output.contiguous().view(self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim) # reshape to normal dim

        attn_mask = get_attn_mask(ngh_msk, self.time_window, neighbors_output.size(), self.use_cuda)


        # node_output = nn.functional.relu(node_output)
        # neighbors_output = nn.functional.relu(neighbors_output)

        node_output = self.dropout(node_output)
        neighbors_output = self.dropout(neighbors_output)
        output, attention = self.attention(node_output, neighbors_output, attn_mask)
        output = self.prj(output)

        output = output.squeeze()
        return output, attention

    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                torch.nn.init.xavier_normal(p.data)
        self.attention.layer_norm.gamma.data.fill_(1)

class JordanRNNJointAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, n_head, nlayers, max_neighbors, time_steps, time_window, dropout_prob=0.1, temperature=1):
        super(JordanRNNJointAttention, self).__init__()
        self.NodeRNN = nn.GRUCell(input_dim+3, hidden_dim)
        self.NeighborRNN = nn.GRUCell(input_dim+3, hidden_dim)
        self.attention = FeatureTransformerLayer(n_head, hidden_dim, max_neighbors, time_steps, temperature)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        self.dropout = nn.Dropout(dropout_prob)
            # nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
            #                      nn.Tanh(),
            #                      nn.Dropout(dropout_prob),
            #                      nn.Linear(hidden_dim // 2, output_dim))

        self.name = "Jordan_RNN" + self.attention.name


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.time_window = time_window
        self.n_ead = n_head


    def forward(self, node_input, neighbors_input, ngh_msk, node_hidden, neighbors_hidden, target):
        """
        1) compute node RNN
        2) compute neighbors RNN
        3) compute attentions
        4) apply dropout on attention

        :param node_input: node input
        :param node_hidden: hidden state for node rnn
        :param neighbors_input: neighbors input
        :param neighbors_hidden: hidden state for neighbors rnn
        :param ngh_msk: number of neighbors
        :return:
        """
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        flat_neighbors_input = neighbors_input.view(-1, self.time_steps, self.input_dim)
        node_output = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, self.hidden_dim).zero_())
        neighbors_output = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim).zero_())
        outputs = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, 1).zero_())
        conditional_outputs = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, 3).zero_())
        attentions = TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, (self.max_neighbors+1) * self.time_steps).zero_()

        attn_mask = get_attn_mask(ngh_msk, self.time_window, neighbors_output.size(), self.use_cuda)

        for i in range(self.time_steps):
            node_hidden = self.NodeRNN(torch.cat((node_input[:, i], conditional_outputs[:, i]), dim=-1), node_hidden)
            neighbors_hidden = self.NeighborRNN(torch.cat((flat_neighbors_input[:, i],
                       conditional_outputs[:, i].unsqueeze(1).expand(-1, self.max_neighbors, -1).contiguous().view(-1, 3)),
                      dim=-1), neighbors_hidden)

            # node_hidden = nn.functional.relu(node_hidden)
            # neighbors_hidden = nn.functional.relu(neighbors_hidden)

            node_hidden = self.dropout(node_hidden)
            neighbors_hidden = self.dropout(neighbors_hidden)

            node_output[:, i] = node_hidden
            neighbors_output[:, :, i] = neighbors_hidden.contiguous().view(self.batch_size, self.max_neighbors, self.hidden_dim) # reshape to normal dim


            output, attention = self.attention(node_output, neighbors_output, i, attn_mask)

            output = self.prj(output)
            outputs[:, i] = output
            attentions[:, i] = attention.squeeze()

            upper_bound = i+1
            lower_bound = i-2 if i >= 3 else 0
            diff = (outputs[:, lower_bound:upper_bound, 0] - target[:, lower_bound:upper_bound, 0])
            if i < 2:
                conditional_outputs[:, i+1] = torch.cat((diff, Variable(TENSOR_TYPE["f_tensor"](self.batch_size, 3-(i+1)).zero_())), dim=-1)
            elif i < self.time_steps-1:
                conditional_outputs[:, i+1] = diff

        return outputs, attentions

    def reset_parameters(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                torch.nn.init.xavier_normal(p.data)
        self.attention.layer_norm.gamma.data.fill_(1)

    def init_hidden(self, batch_size):
        """
        generate a new hidden state to avoid the back-propagation to the beginning to the dataset
        :return:
        """
        hidden = torch.autograd.Variable(torch.zeros((batch_size, self.hidden_dim)))
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        return hidden


class RNNJordanJointAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, time_steps, time_window, dropout_prob=0.1):
        super(RNNJordanJointAttention, self).__init__()
        self.NodeRNN = nn.GRUCell(input_dim, hidden_dim)
        self.NeighborRNN = nn.GRUCell(input_dim, hidden_dim)
        self.Attention = FeatureJointAttention(hidden_dim, max_neighbors, time_steps)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 # nn.Linear(hidden_dim // 2, output_dim),
                                 nn.ReLU())

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


    def forward(self, node_input, neighbors_input, ngh_msk, node_hidden, neighbors_hidden, target):
        """
        1) compute node RNN
        2) compute neighbors RNN
        3) compute attentions
        4) apply dropout on attention

        :param node_input: node input
        :param node_hidden: hidden state for node rnn
        :param neighbors_input: neighbors input
        :param neighbors_hidden: hidden state for neighbors rnn
        :param ngh_msk: number of neighbors
        :return:
        """
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        flat_neighbors_input = neighbors_input.view(-1, self.time_steps, self.input_dim)
        node_output = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, self.hidden_dim).zero_())
        neighbors_output = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim).zero_())
        outputs = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, 1).zero_())
        conditional_outputs = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, 3).zero_())
        attentions = TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, self.max_neighbors+1, self.time_steps).zero_()
        att_norms = []

        atten_mask = get_attn_mask(neighbors_output.size(), self.use_cuda, self.time_window)

        for i in range(self.time_steps):
            node_hidden = self.NodeRNN(node_input[:, i], node_hidden)
            neighbors_hidden = self.NeighborRNN(flat_neighbors_input[:, i], neighbors_hidden)

            node_output[:, i] = node_hidden
            neighbors_output[:, :, i] = neighbors_hidden.contiguous().view(self.batch_size, self.max_neighbors, self.hidden_dim) # reshape to normal dim


            output, attention, att_norm = self.Attention(node_output, neighbors_output, conditional_outputs, ngh_msk, i, atten_mask[:, i])
            att_norms.append(att_norm)

            output = self.dropout(output)
            output = self.prj(output)
            outputs[:, i] = output
            attentions[:, i] = attention

            upper_bound = i+1
            lower_bound = i-2 if i >= 3 else 0
            diff = (outputs[:, lower_bound:upper_bound, 0] - target[:, lower_bound:upper_bound, 0])
            if i < 2:
                conditional_outputs[:, i+1] = torch.cat((diff, Variable(TENSOR_TYPE["f_tensor"](self.batch_size, 3-(i+1)).zero_())), dim=-1)
            elif i < self.time_steps-1:
                conditional_outputs[:, i+1] = diff

            # conditional_outputs[:, i, 0:min(upper_bound, 3)] =

        # outputs = self.dropout(outputs)
        # outputs = self.prj(outputs)
        return outputs, attentions, torch.mean(torch.mean(torch.stack(att_norms, dim=0), dim=0))

    def init_hidden(self, batch_size):
        """
        generate a new hidden state to avoid the back-propagation to the beginning to the dataset
        :return:
        """
        hidden = Variable(torch.zeros((batch_size, self.hidden_dim)))
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        return hidden



class OnlyJointAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, time_steps, time_window, dropout_prob=0.1):
        super(OnlyJointAttention, self).__init__()

        self.node_prj = nn.Linear(input_dim, hidden_dim)
        self.neighbors_prj = nn.Linear(input_dim, hidden_dim)

        self.Attention = JointAttention(hidden_dim, max_neighbors, time_steps)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim // 2, output_dim),
                                 nn.ELU())

        self.name = "Only" + self.Attention.name
        self.dropout = nn.Dropout(dropout_prob)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.time_window = time_window

    def forward(self, node_input, neighbors_input, ngh_msk):
        """
        3) compute attentions
        4) apply dropout on attention

        :param node_input: node input
        :param node_hidden: hidden state for node rnn
        :param neighbors_input: neighbors input
        :param neighbors_hidden: hidden state for neighbors rnn
        :param ngh_msk: number of neighbors
        :return:
        """
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        flat_neighbors_input = neighbors_input.view(-1, self.input_dim)
        node_output = self.node_prj(node_input.view(-1, self.input_dim)).contiguous().view(self.batch_size, self.time_steps, -1)
        neighbors_output = self.neighbors_prj(flat_neighbors_input).contiguous().view(self.batch_size, self.max_neighbors, self.time_steps, -1)

        atten_mask = get_attn_mask(neighbors_output.size(), self.use_cuda, self.time_window)

        output, attention, att_norms = self.Attention(node_output, neighbors_output, ngh_msk, atten_mask)
        output = self.dropout(output)
        output = self.prj(output)
        return output, attention, torch.mean(att_norms, dim=0)[0]



class JordanJointAttention(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, nlayers, max_neighbors, time_steps, time_window, dropout_prob=0.1):
        super(JordanJointAttention, self).__init__()

        self.node_prj = nn.Linear(input_dim, hidden_dim)
        self.neighbors_prj = nn.Linear(input_dim, hidden_dim)

        self.Attention = FeatureJointAttention(hidden_dim, max_neighbors, time_steps)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim // 2, output_dim),
                                 nn.ELU())

        self.name = "Jordan" +self.Attention.name
        self.dropout = nn.Dropout(dropout_prob)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = nlayers
        self.time_steps = time_steps
        self.max_neighbors = max_neighbors
        self.time_window = time_window

    def forward(self, node_input, neighbors_input, ngh_msk, target):
        """
        3) compute attentions
        4) apply dropout on attention

        :param node_input: node input
        :param node_hidden: hidden state for node rnn
        :param neighbors_input: neighbors input
        :param neighbors_hidden: hidden state for neighbors rnn
        :param ngh_msk: number of neighbors
        :return:
        """
        self.batch_size = node_input.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        node_output = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, self.hidden_dim).zero_())
        neighbors_output = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim).zero_())
        outputs = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, 1).zero_())
        conditional_outputs = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, 3).zero_())
        attentions = TENSOR_TYPE["f_tensor"](self.batch_size, self.time_steps, self.max_neighbors + 1, self.time_steps).zero_()
        att_norms = []


        for i in range(self.time_steps):
            flat_neighbors_input = neighbors_input[:, :, i].view(-1, self.input_dim)
            node_output[:, i] = self.node_prj(node_input[:, i])
            neighbors_output[:, :, i] = self.neighbors_prj(flat_neighbors_input).contiguous().view(self.batch_size, self.max_neighbors, -1)

            atten_mask = get_attn_mask(neighbors_output.size(), self.use_cuda, self.time_window)

            # attend to current row
            if i < self.time_steps - 1:
                atten_mask[:, :, i + 1:] = 1
                atten_mask[:, i + 1:] = 1
                if i > 0:
                    atten_mask[:, :i] = 1

            output, attention, att_norm = self.Attention(node_output, neighbors_output, conditional_outputs, ngh_msk, i, atten_mask)
            att_norms.append(att_norm)

            output = self.dropout(output)
            output = self.prj(output)
            outputs[:, i] = output
            attentions[:, i] = attention

            upper_bound = i + 1
            lower_bound = i - 2 if i >= 3 else 0
            diff = (outputs[:, lower_bound:upper_bound, 0] - target[:, lower_bound:upper_bound, 0])
            if i < 2:
                conditional_outputs[:, i + 1] = torch.cat((diff, Variable(TENSOR_TYPE["f_tensor"](self.batch_size, 3 - (i + 1)).zero_())),
                                                          dim=-1)
            elif i < self.time_steps - 1:
                conditional_outputs[:, i + 1] = diff

        return outputs, attentions, torch.mean(torch.mean(torch.stack(att_norms, dim=0), dim=0))