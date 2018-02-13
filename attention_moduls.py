from torch import nn
import torch
from helper import get_temperature, TENSOR_TYPE


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
        self.W = nn.Parameter(TENSOR_TYPE["f_tensor"](in_out_dim, in_out_dim))
        self.transpose = transpose
        if baias_size > 0:
            self.b = nn.Parameter(TENSOR_TYPE["f_tensor"](baias_size))


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

        outputs = torch.autograd.Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.max_neighbors+1, self.time_steps, self.hidden_dim).zero_())
        atts = TENSOR_TYPE["f_tensor"](self.batch_size, self.max_neighbors + 1, self.time_steps, self.time_steps).zero_()
        # outputs = Variable(TENSOR_TYPE["f_tensor"](self.batch_size, self.max_neighbors, self.time_steps, self.hidden_dim).zero_())
        # atts = TENSOR_TYPE["f_tensor"](self.batch_size, self.max_neighbors, self.time_steps, self.time_steps).zero_()


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


class JointAttention(nn.Module):
    def __init__(self, hidden_dim, max_neighbors, time_steps, temperature=1):
        super(JointAttention, self).__init__()
        self.name = "_JointAttention"

        self.softmax = nn.Softmax(dim=-1)

        # self.proj = BiLinearProjection(hidden_dim, time_steps)
        # self.proj_v = BiLinearProjection((max_neighbors + 1) * time_steps, hidden_dim, transpose=False)

        self.proj_query = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.proj_key = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.proj_value = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.init_params()
        self.temperature = temperature
    def init_params(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_normal(p.data)

    def forward(self, node_output, neigh_output, neighbors_number, attn_mask=None):
        self.batch_size = node_output.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        querys = node_output.repeat(self.max_neighbors + 1, 1, 1)
        key_values = torch.cat((node_output, torch.cat(torch.split(neigh_output, 1, dim=1), dim=0).squeeze()), dim=0)

        w_querys = self.proj_query(querys)
        w_key = self.proj_key(key_values)
        w_values = self.proj_value(torch.cat((node_output.unsqueeze(1), neigh_output), dim=1).view(self.batch_size, -1, self.hidden_dim))

        S = torch.bmm(w_querys, w_key.transpose(1, 2))
        S = torch.stack(torch.split(S, self.batch_size, dim=0), dim=1).squeeze()
        # S = (S / self.hidden_dim**0.5) * (self.max_neighbors+1)
        S_norm = torch.norm(S.data.contiguous().view(self.batch_size, -1), 2, -1)

        if attn_mask is not None:
            S.data.masked_fill_(attn_mask.unsqueeze(1).expand(-1, self.max_neighbors + 1, -1, -1), -float('inf'))
        S = S.transpose(1, 2)

        S /= self.temperature
        A = self.softmax(S.contiguous().view(self.batch_size, self.time_steps, (self.max_neighbors + 1) * self.time_steps))
        output = torch.bmm(A, w_values.view(self.batch_size, -1, self.hidden_dim))
        return output, A.data.view(self.batch_size, self.time_steps, self.max_neighbors + 1, -1), S_norm




class FeatureJointAttention(nn.Module):
    def __init__(self, hidden_dim, max_neighbors, time_steps, temperature=1):
        super(FeatureJointAttention, self).__init__()
        self.name = "_FeatureJointAttention"

        self.softmax = nn.Softmax(dim=-1)
        self.proj_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.proj_node_output = nn.Linear(hidden_dim + 3, hidden_dim)

        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.temperature = temperature
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_normal(p.data)

    def forward(self, node_output, neigh_output, neighbors_number, input_time_steps, attn_mask=None):
        self.batch_size = node_output.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        querys = node_output.repeat(self.max_neighbors+1, 1, 1)
        key_values = torch.cat((node_output, torch.cat(torch.split(neigh_output, 1, dim=1), dim=0).squeeze()), dim=0)

        w_querys = self.proj_query(querys)
        w_key = self.proj_key(key_values)
        w_values = self.proj_value(torch.cat((node_output.unsqueeze(1), neigh_output), dim=1).view(self.batch_size, -1, self.hidden_dim))

        S = torch.bmm(w_querys, w_key.transpose(1, 2))
        S = torch.stack(torch.split(S, self.batch_size), dim=1)
        S = S[:, :, input_time_steps]
        S_norm = torch.norm(S.data.contiguous().view(self.batch_size, -1), 2, -1)

        if attn_mask is not None:
            S.data.masked_fill_(attn_mask.unsqueeze(1).expand(-1, self.max_neighbors+1, -1), -float('inf'))
        S = S.contiguous().view(self.batch_size, 1, -1)

        A = self.softmax(S)
        output = torch.bmm(A, w_values).squeeze()
        return output, A.data.view(self.batch_size, self.max_neighbors+1, -1), S_norm

