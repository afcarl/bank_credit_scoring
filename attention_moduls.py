from torch import nn
import torch
import torch.nn.init as init

from helper import get_temperature, TENSOR_TYPE, LayerNorm, BiLinearProjection, hookFunc, GumbelSoftmax, TempSoftmax

class EdgeEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, temperature=0.77, dropout=0):
        super(EdgeEncoder, self).__init__()

        self.EdgeGRU = nn.GRU(input_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
        self.prj = nn.Sequential(nn.Linear(hidden_dim, out_dim),
                                 nn.Dropout(dropout),
                                 nn.ELU())
        self.attention = GumbelSoftmax(temperature)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, node_in, neigh_in):
        """
        Compute a soft edge type
        :param neigh_in:
        :return:
        """
        batch_size, max_negih, time_steps, hidden_dim = neigh_in.size()


        hidden = torch.autograd.Variable(torch.zeros((1, max_negih*batch_size, self.hidden_dim)))
        if torch.cuda.is_available():
            hidden = hidden.cuda()

        # reshape
        node_in = node_in.repeat(max_negih, 1, 1)
        neigh_in = torch.cat(torch.split(neigh_in, 1, dim=1), dim=0)[:, 0]
        # create independent encoding
        edges_enc = torch.cat((node_in, neigh_in), dim=-1)
        edges_enc = edges_enc.bmm(edges_enc.transpose(1, 2))

        # get edge encoding overtime
        edge_enc_out, hidden = self.EdgeGRU(edges_enc, hidden)
        edge_enc_out = self.prj(edge_enc_out)

        # get edge type distribution
        edge_enc_out /= self.temperature
        output = self.attention(edge_enc_out)
        output = torch.stack(torch.split(output, batch_size, dim=0), dim=1)
        return output






class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, hidden_dim, max_neighbors, time_steps, temperature=1, dropout=0):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.temperature = temperature
        self.softmax = TempSoftmax(temperature)
        # self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.prj = nn.Linear(hidden_dim * n_head, hidden_dim)



    def forward(self, q, k, v, batch_size, attn_mask=None):
        S = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            S.data.masked_fill_(attn_mask, -float('inf'))

        # S /= (self.temperature * (self.hidden_dim ** 0.5))
        # S /= self.temperature
        A = self.softmax(S)
        output = torch.bmm(A, v)
        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)
        output = self.dropout(output)
        output = self.prj(output)
        self.att_m = output
        return output, torch.stack(torch.split(A.data, batch_size, dim=0), dim=0).sum(0)



class FeatureMultiHeadAttention(nn.Module):
    def __init__(self, n_head, hidden_dim, max_neighbors, time_steps, temperature=1, dropout=0):
        super(FeatureMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.temperature = temperature

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.prj = nn.Linear(hidden_dim * n_head, hidden_dim)


    def forward(self, q, k, v, batch_size, attn_mask=None):
        S = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            S.data.masked_fill_(attn_mask, -float('inf'))

        S /= (self.temperature * (self.hidden_dim ** 0.5))
        A = self.softmax(S)
        output = torch.bmm(A, v).squeeze()
        output = torch.cat(torch.split(output, batch_size, dim=0), dim=-1)
        output = self.dropout(output)
        output = self.prj(output)
        return output, torch.stack(torch.split(A.data, batch_size, dim=0), dim=0).sum(0)


class TransformerLayer(nn.Module):
    def __init__(self, n_head, input_dim, hidden_dim, component_num, time_steps, temperature=1, dropout=0):
        super(TransformerLayer, self).__init__()
        self.name = "_TransformerAttention"
        self.n_head = n_head
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.component_num = component_num
        self.time_steps = time_steps


        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, input_dim, hidden_dim))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, input_dim, hidden_dim))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, input_dim, hidden_dim))

        self.slf_attn = MultiHeadAttention(n_head, hidden_dim, component_num, time_steps, temperature=temperature, dropout=dropout)
        self.layer_norm = LayerNorm(hidden_dim)



    def forward(self, node_enc, neigh_enc, attn_mask=None):
        batch_size = node_enc.size(0)

        q = node_enc
        k = torch.cat((node_enc, torch.cat(torch.split(neigh_enc, 1, dim=1), dim=2)[:, 0]), dim=1)
        v = torch.cat((node_enc.unsqueeze(1), neigh_enc), dim=1).view(batch_size, -1, self.input_dim)

        q_s = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.input_dim)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.input_dim)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.input_dim)  # n_head x (mb_size*len_v) x d_model

        q_s = torch.bmm(q_s, self.w_qs).view(-1, self.time_steps, self.hidden_dim)  # (n_head*mb_size) x seq_le x hidden_dim
        k_s = torch.bmm(k_s, self.w_ks).view(self.n_head * batch_size, -1, self.hidden_dim)  # n_head*batch_size, max_neighbors+1*seq_le, hidden_dim
        v_s = torch.bmm(v_s, self.w_vs).view(self.n_head * batch_size, -1, self.hidden_dim)  # n_head*batch_size, max_neighbors+1*seq_le, hidden_dim

        output, slf_attn = self.slf_attn(q_s, k_s, v_s, batch_size, attn_mask=attn_mask.repeat(self.n_head, 1, 1))
        output = self.layer_norm(output + node_enc)
        # output = output + node_enc

        # output = self.pos_ffn(output)


        return output, slf_attn



class FeatureTransformerLayer(nn.Module):
    def __init__(self, n_head, hidden_dim, max_neighbors, time_steps, temperature=1, dropout=0):
        super(FeatureTransformerLayer, self).__init__()
        self.name = "_FeatureTransformerAttention"
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps


        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, hidden_dim, hidden_dim))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, hidden_dim, hidden_dim))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, hidden_dim, hidden_dim))

        self.slf_attn = FeatureMultiHeadAttention(n_head, hidden_dim, max_neighbors, time_steps, temperature=temperature, dropout=dropout)
        self.layer_norm = LayerNorm(hidden_dim)




    def forward(self, node_enc, neigh_enc, current_time_step, attn_mask=None):
        batch_size = node_enc.size(0)

        q = node_enc[:, current_time_step].unsqueeze(1)
        k = torch.cat((node_enc, torch.cat(torch.split(neigh_enc, 1, dim=1), dim=2).squeeze()), dim=1)
        v = torch.cat((node_enc, torch.cat(torch.split(neigh_enc, 1, dim=1), dim=2).squeeze()), dim=1)

        q_s = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.hidden_dim)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.hidden_dim)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.hidden_dim)  # n_head x (mb_size*len_v) x d_model

        q_s = torch.bmm(q_s, self.w_qs).view(-1, 1, self.hidden_dim)  # (n_head*mb_size*max_neighbors+1) x seq_le x hidden_dim
        k_s = torch.bmm(k_s, self.w_ks).view(self.n_head * batch_size, -1,
                                             self.hidden_dim)  # (n_head*mb_size*max_neighbors+1) x seq_le x hidden_dim
        v_s = torch.bmm(v_s, self.w_vs).view(self.n_head * batch_size, -1, self.hidden_dim)  # n_head*batch_size, max_neighbors+1*seq_le, hidden_dim

        attn_mask[:, current_time_step]
        attn_mask = attn_mask.unsqueeze(1).repeat(self.n_head, 1, 1)

        output, slf_attn = self.slf_attn(q_s, k_s, v_s, batch_size, attn_mask=attn_mask)
        output = self.layer_norm(output + node_enc[:, current_time_step])
        # output = self.pos_ffn(output)


        return output, slf_attn



# class TransformerJointAttention(nn.Module):
#     def __init__(self, n_head, hidden_dim, max_neighbors, time_steps, n_layer=3, temperature=1, dropout=0.1):
#         super(TransformerJointAttention, self).__init__()
#         self.name = "_TransformerJointAttention"
#         self.n_head = n_head
#         self.hidden_dim = hidden_dim
#         self.max_neighbors = max_neighbors
#         self.time_steps = time_steps
#
#         self.layer_stack = nn.ModuleList([TransformerLayer(n_head, hidden_dim, max_neighbors, time_steps, temperature, dropout=dropout)
#             for _ in range(n_layer)])
#
#
#     def forward(self, node_enc, neighbors_enc, attn_mask=None):
#         enc_output = torch.cat((node_enc.unsqueeze(1), neighbors_enc), dim=1)
#         return_attns = []
#
#         for enc_layer in self.layer_stack:
#             enc_output, enc_slf_attn = enc_layer(enc_output, attn_mask)
#             return_attns += [enc_slf_attn]
#
#         return enc_output, return_attns



class NetAttention(nn.Module):
    def __init__(self, n_head, hidden_dim, max_neighbors, time_steps, temperature=1, drop_prob=0.1):
        super(NetAttention, self).__init__()
        self.name = "_NetAttention"

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_prob)
        self.prj = nn.Linear(hidden_dim * n_head, hidden_dim)

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, hidden_dim, hidden_dim))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, hidden_dim, hidden_dim))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, hidden_dim, hidden_dim))


        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.n_head = n_head
        self.temp = temperature

    def forward(self, node_output, neigh_output, att_mask):
        self.batch_size = node_output.size(0)

        q = node_output
        k = torch.cat((node_output, torch.cat(torch.split(neigh_output, 1, dim=1), dim=2).squeeze()), dim=1)
        v = torch.cat((node_output.unsqueeze(1), neigh_output), dim=1).view(self.batch_size, -1, self.hidden_dim)

        q_s = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.hidden_dim)  # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.hidden_dim)  # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.hidden_dim)  # n_head x (mb_size*len_v) x d_model

        q_s = torch.bmm(q_s, self.w_qs).view(-1, self.time_steps, self.hidden_dim)  # (n_head*mb_size*max_neighbors+1) x seq_le x hidden_dim
        k_s = torch.bmm(k_s, self.w_ks).view(self.n_head * self.batch_size, -1,
                                             self.hidden_dim)  # (n_head*mb_size*max_neighbors+1) x seq_le x hidden_dim
        v_s = torch.bmm(v_s, self.w_vs).view(self.n_head * self.batch_size, -1,
                                             self.hidden_dim)  # n_head*batch_size, max_neighbors+1*seq_le, hidden_dim


        S = torch.bmm(q_s, k_s.transpose(1, 2))
        S.data.masked_fill_(att_mask.repeat(self.n_head, 1, self.max_neighbors+1), -float('inf'))
        S /= self.temp

        A = self.softmax(S)
        output = torch.bmm(A, v_s)
        output = torch.cat(torch.split(output, self.batch_size, dim=0), dim=-1)
        output = self.prj(output)
        return output, torch.stack(torch.split(A.data, self.batch_size, dim=0), dim=0).sum(0)

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


class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_dim, max_neighbors, time_steps, temperature=1, dropout=0):
        super(SingleHeadAttention, self).__init__()
        self.name = "_SingleHeadAttention"

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # self.proj = BiLinearProjection(hidden_dim, time_steps)
        # self.proj_v = BiLinearProjection((max_neighbors + 1) * time_steps, hidden_dim, transpose=False)

        self.proj_query = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.proj_key = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.proj_value = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))

        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.temperature = temperature


    def forward(self, node_output, neigh_output, attn_mask=None):
        self.batch_size = node_output.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        querys = node_output.repeat(self.max_neighbors + 1, 1, 1)
        keys = torch.cat((node_output, torch.cat(torch.split(neigh_output, 1, dim=1), dim=0).squeeze()), dim=0)
        values = torch.cat((node_output.unsqueeze(1), neigh_output), dim=1).view(self.batch_size, -1, self.hidden_dim)

        w_querys = querys.matmul(self.proj_query)
        w_key = keys.matmul(self.proj_key)
        w_values = values.matmul(self.proj_value)

        S = torch.bmm(w_querys, w_key.transpose(1, 2))
        S = torch.cat(torch.split(S, self.batch_size, dim=0), dim=-1)

        if attn_mask is not None:
            S.data.masked_fill_(attn_mask.repeat(1, 1, self.max_neighbors+1), -float('inf'))

        S /= self.temperature
        A = self.softmax(S)
        A = self.dropout(A)
        output = torch.bmm(A, w_values)
        return output, A.data.view(self.batch_size, self.time_steps, self.max_neighbors + 1, -1)




class FeatureSingleHeadAttention(nn.Module):
    def __init__(self, hidden_dim, max_neighbors, time_steps, temperature=1):
        super(FeatureSingleHeadAttention, self).__init__()
        self.name = "_FeatureSingleHeadAttention"

        self.softmax = nn.Softmax(dim=-1)
        self.proj_query = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.proj_key = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.proj_value = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))

        # self.proj_node_output = nn.Linear(hidden_dim + 3, hidden_dim)

        self.hidden_dim = hidden_dim
        self.max_neighbors = max_neighbors
        self.time_steps = time_steps
        self.temperature = temperature


    def forward(self, node_output, neigh_output, neighbors_number, current_time_step, attn_mask=None):
        self.batch_size = node_output.size(0)
        self.use_cuda = next(self.parameters()).is_cuda

        querys = node_output.repeat(self.max_neighbors + 1, 1, 1)
        keys = torch.cat((node_output, torch.cat(torch.split(neigh_output, 1, dim=1), dim=0).squeeze()), dim=0)
        values = torch.cat((node_output.unsqueeze(1), neigh_output), dim=1).view(self.batch_size, -1, self.hidden_dim)

        w_querys = querys.matmul(self.proj_query)
        w_key = keys.matmul(self.proj_key)
        w_values = values.matmul(self.proj_value)

        S = torch.bmm(w_querys, w_key.transpose(1, 2))
        S = torch.cat(torch.split(S, self.batch_size, dim=0), dim=-1)
        S = S[:, current_time_step]

        if attn_mask is not None:
            S.data.masked_fill_(attn_mask.repeat(1, self.max_neighbors+1), -float('inf'))

        S /= self.temperature
        A = self.softmax(S)
        output = torch.bmm(A.unsqueeze(1), w_values).squeeze()
        return output, torch.stack(torch.split(A.data, self.time_steps, dim=-1), dim=1)

