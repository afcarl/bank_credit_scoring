import torch
from helper import rmse, TIMESTAMP, get_attn_mask, get_temperature
from torch.autograd import Variable
import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
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


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class BiLinearProjection(nn.Module):
    def __init__(self, in_out_dim, baias_size, transpose=True):
        super(BiLinearProjection, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(in_out_dim, in_out_dim))
        self.transpose = transpose
        if baias_size > 0:
            self.b = nn.Parameter(torch.FloatTensor(baias_size))


    def init_params(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_normal(p.data)

    def forward(self, node, neighbor):
        output = node.matmul(self.W)
        if self.transpose:
            output = output.matmul(neighbor.transpose(1, 2))
        else:
            output = output.matmul(neighbor)
        return output + self.b


class NetAttention(nn.Module):
    def __init__(self, n_head, hidden_dim, max_neighbors, time_steps, drop_prob=0.1):
        super(NetAttention, self).__init__()
        self.name = "_NetAttention"

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_prob)
        self.proj = BiLinearProjection(hidden_dim*time_steps, max_neighbors)

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps

        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_normal(p.data)

    def forward(self, node_rnn_output, neigh_rnn_output, neighbors_number):
        self.batch_size = node_rnn_output.size(0)

        stacked_node_rnn_output = node_rnn_output.repeat(self.max_neighbors, 1, 1).view(self.batch_size, self.max_neighbors, -1)              # (n_head*b_size, #_neighbor, time * hidden_dim)
        flat_neigh_rnn_output = neigh_rnn_output.view(self.batch_size, self.max_neighbors, -1)    # (n_head*b_size, #_neighbor, time * hidden_dim)

        S = self.proj(stacked_node_rnn_output, flat_neigh_rnn_output)
        A = self.softmax(S)
        A = self.dropout(A)
        output = torch.bmm(A, flat_neigh_rnn_output)

        return output.view(self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim), \
               torch.stack(torch.split(A, self.batch_size, dim=0), dim=1)

class TimeAttention(nn.Module):
    def __init__(self, n_head, hidden_dim, max_neighbors, time_steps, drop_prob=0.1):
        super(TimeAttention, self).__init__()
        self.name = "_TimeAttention"

        # self.softmax = nn.Sequential(nn.Tanh(), nn.Softmax(dim=-1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_prob)

        self.proj = BiLinearProjection(hidden_dim, time_steps)
        self.proj_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_value = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_normal(p.data)

    def forward(self, node_rnn_output, neigh_rnn_output, neighbors_number, attn_mask=None):
        self.batch_size = node_rnn_output.size(0)
        self.use_cuda = next(self.parameters()).is_cuda


        querys = node_rnn_output.unsqueeze(1).expand(-1, self.max_neighbors+1, -1, -1)
        key_values = torch.cat((node_rnn_output.unsqueeze(1), neigh_rnn_output), dim=1)
        # querys = node_rnn_output.unsqueeze(1).expand(-1, self.max_neighbors, -1, -1)
        # key_values = neigh_rnn_output

        outputs = Variable(torch.FloatTensor(self.batch_size, self.max_neighbors+1, self.time_steps, self.hidden_dim).zero_())
        atts = torch.FloatTensor(self.batch_size, self.max_neighbors + 1, self.time_steps, self.time_steps).zero_()
        # outputs = Variable(torch.FloatTensor(self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim).zero_())
        # atts = torch.FloatTensor(self.batch_size, self.max_neighbors, self.time_steps, self.time_steps).zero_()

        if self.use_cuda:
            outputs = outputs.cuda()
            atts = atts.cuda()

        for i in range(self.max_neighbors+1):
        # for i in range(self.max_neighbors):
            query = self.proj_query(querys[:, i])
            key = self.proj_key(key_values[:, i])
            value = self.proj_value(key_values[:, i])

            S = self.proj(query, key)
            # S = key
            if attn_mask is not None:
                S.data.masked_fill_(attn_mask, -float('inf'))
            A = self.softmax(S)
            A = self.dropout(A)
            output = torch.bmm(A, value)
            outputs[:, i] = output
            atts[:, i] = A.data
        return outputs, atts

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
        self.n_layers = nlayers
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
        self.n_layers = nlayers
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


class JointAttention(nn.Module):
    def __init__(self, hidden_dim, max_neighbors, time_steps, drop_prob=0.1):
        super(JointAttention, self).__init__()
        self.name = "_JointAttention"

        self.softmax = nn.Softmax(dim=-1)

        # self.proj_s = NodeNeighborProjection(hidden_dim, time_steps)
        self.proj = BiLinearProjection(hidden_dim, time_steps)
        self.proj_v = BiLinearProjection((max_neighbors + 1) * time_steps, hidden_dim, transpose=False)

        # self.proj_query = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.proj_key = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.proj_value = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_normal(p.data)

    def forward(self, node_rnn_output, neigh_rnn_output, neighbors_number, attn_mask=None):
        self.batch_size = node_rnn_output.size(0)
        self.use_cuda = next(self.parameters()).is_cuda


        querys = node_rnn_output.unsqueeze(1).expand(-1, self.max_neighbors+1, -1, -1)
        querys = querys.contiguous().view(-1, self.time_steps, self.hidden_dim)
        key_values = torch.cat((node_rnn_output.unsqueeze(1), neigh_rnn_output), dim=1)
        key_values = key_values.view(-1, self.time_steps, self.hidden_dim)

        # w_querys = self.proj_query(querys)
        # w_key = self.proj_key(key_values)
        # w_values = self.proj_value(key_values)

        S = self.proj(querys, key_values)
        # S = torch.bmm(w_querys, w_key.transpose(1,2))
        S = S.view(self.batch_size, self.max_neighbors+1, self.time_steps, self.time_steps)

        if attn_mask is not None:
            S.data.masked_fill_(attn_mask.unsqueeze(1).expand(-1, self.max_neighbors+1, -1, -1), -float('inf'))
        S = S.transpose(1, 2)


        # s = [self.proj(querys[:, i], key_values[:, i]) for i in range(self.max_neighbors+1)]
        # s = [m.data.masked_fill_(attn_mask, -float('inf')) for m in s]
        # s_cat = torch.stack(s, dim=2)
        #
        # assert (S.data == s_cat).any()

        A = self.softmax(S.contiguous().view(self.batch_size, self.time_steps, (self.max_neighbors+1)*self.time_steps))
        output = self.proj_v(A, key_values.view(self.batch_size, -1, self.hidden_dim))
        # output = torch.bmm(A, w_values.view(self.batch_size, -1, self.hidden_dim))
        # output = torch.bmm(A, key_values.view(self.batch_size, -1, self.hidden_dim))
        return output, A.data.view(self.batch_size, self.time_steps, self.max_neighbors+1, -1)


class ClusteringExamples(nn.Module):
    def __init__(self, hidden_dim, out_dim, time_steps, decadicy_iteration, max_temp=1, min_tem=1, total_iteration=None):
        super(ClusteringExamples, self).__init__()
        self.name = "_cluster_example"
        self.proj = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.ReLU())
        self.temperature = get_temperature(max_temp, min_tem, decadicy_iteration, total_iteration)
        self.select = nn.Softmax(dim=-1)


        self.hidden_dim = hidden_dim
        self.time_steps = time_steps

    def forward(self, input, n_iter, top_k=1):
        input = self.proj(input.view(-1, self.hidden_dim))
        input /= self.temperaturep[n_iter]

        prob = self.select(input)
        max_value, selected_element = torch.max(prob)
        prob[:] = 0
        prob[selected_element] = 1

        return prob


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
        self.n_layers = nlayers
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


