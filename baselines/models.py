from torch import nn, functional as F
from torch.autograd import Variable
import torch

from helper import rmse, AttrProxy, BaseNet, LayerNorm


class GAT(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, num_neighbours, dropout_prob=0.1):
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = 1
        self.name = "Multi-head GAT"
        self.n_head = 4

        self.node_enc = torch.nn.ParameterList([nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
                                                for i in range(self.n_head)])
        self.a = torch.nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(2*hidden_dim, 1)) for i in range(self.n_head)])

        self.edge_att = nn.Softmax(dim=1)
        self.node_prj = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU())
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim))

    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, edge_types, mask_neight, mask_time):
        batch_size, neigh_number, time_steps, input_dim = neighbors_input.size()
        node_mask = torch.zeros(batch_size, 1).byte().to(node_input.device)


        mask_neight = torch.cat((node_mask, mask_neight), dim=1)
        mask_neight = mask_neight.unsqueeze(-1).unsqueeze(-1).expand(batch_size, neigh_number+1, time_steps, 1)


        x = torch.cat((node_input.unsqueeze(1), neighbors_input), dim=1)
        edge_rep = []
        for i in range(self.n_head):
            node_enc = torch.matmul(node_input, self.node_enc[i])
            neigh_enc = torch.matmul(neighbors_input, self.node_enc[i])
            neigh_enc = torch.cat((node_enc.unsqueeze(1), neigh_enc), dim=1)

            edges_enc = torch.cat((node_enc.unsqueeze(1).repeat(1, neigh_number+1, 1, 1), neigh_enc), dim=-1)
            edges_enc = torch.matmul(edges_enc, self.a[i])
            edges_enc = edges_enc.data.masked_fill_(mask_neight, -float('inf'))

            edges_att = self.edge_att(edges_enc)
            edge_rep.append(edges_att * neigh_enc)


        node_rep = self.node_prj(torch.cat(edge_rep, dim=-1).sum(dim=1))
        output = self.prj(node_rep)
        return output



class RNNGAT(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, num_neighbours, dropout_prob=0.1):
        super(RNNGAT, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = 1
        self.name = "Recurrent Multi-head GAT"
        self.n_head = 4


        self.rnn_enc = nn.GRU(input_dim, hidden_dim, 1, batch_first=True, bidirectional=False)

        self.node_enc = torch.nn.ModuleList([nn.GRU(input_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
                                                for i in range(self.n_head)])
        self.a = torch.nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(2 * hidden_dim, 1)) for i in range(self.n_head)])

        self.edge_att = nn.Softmax(dim=1)

        self.node_prj = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.ReLU())
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim))


    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, edge_types, mask_neight, mask_time):

        batch_size, neigh_number, time_steps, input_dim = neighbors_input.size()
        node_mask = torch.zeros(batch_size, 1).byte().to(node_input.device)

        mask_neight = torch.cat((node_mask, mask_neight), dim=1)
        mask_neight = mask_neight.unsqueeze(-1).unsqueeze(-1).expand(batch_size, neigh_number + 1, time_steps, 1)
        edge_rep = []

        neigh_inp = torch.cat((node_input.unsqueeze(1), neighbors_input), dim=1)
        neigh_inp = torch.cat(torch.split(neigh_inp, 1, dim=1), dim=0)[:, 0]
        for i in range(self.n_head):
            neigh_enc, _ = self.node_enc[i](neigh_inp, node_hidden[i])
            neigh_enc = torch.stack(torch.chunk(neigh_enc, neigh_number + 1, dim=0), dim=1)
            node_enc = neigh_enc[:, 0]

            edges_enc = torch.cat((node_enc.unsqueeze(1).repeat(1, neigh_number + 1, 1, 1), neigh_enc), dim=-1)
            edges_enc = torch.matmul(edges_enc, self.a[i])
            edges_enc = edges_enc.data.masked_fill_(mask_neight, -float('inf'))

            edges_att = self.edge_att(edges_enc)
            edge_rep.append(edges_att * neigh_enc)

        output = self.node_prj(torch.cat(edge_rep, dim=-1).sum(dim=1))
        output = self.prj(output)
        return output



        batch_size, neigh_number, time_steps, input_dim = neighbors_input.size()
        use_cuda = next(self.parameters()).is_cuda
        node_mask = torch.zeros(batch_size, 1).byte()

        if use_cuda:
            node_mask = node_mask.cuda()

        mask_neight = torch.cat((node_mask, mask_neight), dim=1)
        mask_neight = mask_neight.unsqueeze(-1).unsqueeze(-1).expand(batch_size, neigh_number+1, time_steps, 1).repeat(self.n_head, 1, 1, 1)

        neighbors_input = torch.cat(torch.split(neighbors_input, 1, dim=1), dim=0)[:, 0]

        nodes_enc, node_hidden = self.rnn_enc(torch.cat((node_input, neighbors_input), dim=0), node_hidden)

        node_input = nodes_enc[:batch_size]
        neighbors_input = torch.stack(nodes_enc[batch_size:].split(batch_size, dim=0), dim=1)

        node_input_s = node_input.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.hidden_dim)
        neighbors_input_s = neighbors_input.repeat(self.n_head, 1, 1, 1).view(self.n_head, -1, self.hidden_dim)

        node_enc = torch.bmm(node_input_s, self.node_enc).view(self.n_head * batch_size, time_steps, self.hidden_dim)
        neigh_enc = torch.bmm(neighbors_input_s, self.node_enc).view(self.n_head * batch_size, neigh_number, time_steps, self.hidden_dim)
        neigh_enc = torch.cat((node_enc.unsqueeze(1), neigh_enc), dim=1)


        edges_enc = torch.cat((node_enc.unsqueeze(1).repeat(1, neigh_number+1, 1, 1), neigh_enc), dim=-1)
        edges_enc = torch.bmm(edges_enc.view(self.n_head, -1, 2*self.hidden_dim), self.edge_att_enc).view(self.n_head * batch_size, neigh_number+1, time_steps, 1)
        edges_enc.data.masked_fill_(mask_neight, -float('inf'))

        edges_att = self.edge_att(edges_enc)

        edge_rep = edges_att * neigh_enc
        edge_rep = torch.nn.functional.relu(torch.stack(torch.sum(edge_rep, dim=1).split(batch_size, dim=0), dim=1).mean(dim=1))

        output = self.prj(edge_rep)
        return output


class SimpleGRU(BaseNet):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super(SimpleGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = 1
        self.name = "SimpleRNN"
        self.rnn = nn.GRU(input_dim, hidden_dim,
                          num_layers=1,
                          batch_first=True)
        self.app_rnn = nn.Sequential(nn.Dropout(dropout_prob),
                                     nn.ReLU())

        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())

        # self.prj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
        #                          nn.Tanh(),
        #                          nn.Linear(hidden_dim // 2, output_dim))




    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, edge_types, mask_neight, mask_time):
        """
        forward pass of the network
        :param input: input to the rnn
        :param hidden: hidden state
        :return:
        """
        output, hidden = self.rnn(node_input, node_hidden)
        output = self.app_rnn(output)
        output = self.prj(output)
        return output



class StructuralRNN(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, num_edge_types, dropout_prob=0.1):
        super(StructuralRNN, self).__init__()
        self.NodeRNN = nn.GRU(input_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
        self.app_NodeRNN = nn.Sequential(nn.Dropout(dropout_prob),
                                         nn.ReLU())

        NeighborRNN = [nn.GRU(input_dim, hidden_dim, 1, batch_first=True, bidirectional=False)
                       for i in range(num_edge_types)]

        self.NeighborRNN = nn.ModuleList(NeighborRNN)
        self.app_NeighborRNN = nn.Sequential(nn.Dropout(dropout_prob),
                                             nn.ReLU())

        self.name = "StructuralRNN"
        self.out_RNN = nn.GRU(hidden_dim * (num_edge_types + 1), hidden_dim, 1, batch_first=True, bidirectional=False)
        self.app_outRNN = nn.Sequential(nn.Dropout(dropout_prob),
                                             nn.ReLU())

        self.app_outRNN = nn.Sequential(nn.Dropout(dropout_prob))

        # self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim))

        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())

        # self.prj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
        #                          nn.Tanh(),
        #                          nn.Linear(hidden_dim // 2, output_dim))

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = 1
        self.num_edge_types = num_edge_types
        self.criterion = nn.MSELoss()




    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, edge_types, mask_neight, mask_time):
        use_cuda = next(self.parameters()).is_cuda
        batch_size, neigh_number, time_steps, input_dim = neighbors_input.size()
        out_hidden = self.init_hidden(batch_size)

        out_output = Variable(torch.zeros(batch_size, time_steps, self.hidden_dim * self.num_edge_types))
        if use_cuda:
            out_output = out_output.cuda()
            out_hidden = out_hidden.cuda()

        node_output, node_hidden = self.NodeRNN(node_input, node_hidden)
        # node_output = self.app_NodeRNN(node_output)

        neighbors_hidden = neighbors_hidden.unsqueeze(0).expand(self.num_edge_types, -1, -1, -1)
        edge_types = edge_types.unsqueeze(-1)
        neighbors_input = neighbors_input.unsqueeze(-2) * edge_types

        edge_mask = torch.sum(edge_types, dim=1)
        edge_mask[edge_mask > 1] = 1.
        neighbors_input = torch.sum(neighbors_input, dim=1)


        for t_idx in range(self.num_edge_types):
            output_group_by_type, hidden_group_by_type = self.NeighborRNN[t_idx](neighbors_input[:, :, t_idx], neighbors_hidden[t_idx])
            # output_group_by_type = self.app_NeighborRNN(output_group_by_type)
            output_group_by_type = output_group_by_type * edge_mask[:, :, t_idx]
            out_output[:, :, t_idx * self.hidden_dim:(t_idx + 1) * self.hidden_dim] = output_group_by_type

        output, out_hidden = self.out_RNN(torch.cat((node_output, out_output), dim=-1), out_hidden)
        output = self.app_outRNN(output)
        output = self.prj(output)

        return output


class NodeNeighborsInterpolation(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, num_edge_types, dropout_prob=0.1):
        super(NodeNeighborsInterpolation, self).__init__()
        self.node_prj = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Dropout(dropout_prob),
                                      nn.ReLU())
        self.neight_prj = nn.ModuleList([nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                                       nn.Dropout(dropout_prob),
                                                       nn.ReLU())
                                         for i in range(num_edge_types)])

        # self.prj = nn.Sequential(nn.Linear(hidden_dim * (num_edge_types + 1), hidden_dim),
        #                          nn.Dropout(dropout_prob),
        #                          nn.ELU())


        self.prj = nn.Sequential(nn.Linear(hidden_dim * (num_edge_types + 1), hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())
        self.name = "NodeNeighborsInterpolation"


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_edge_types = num_edge_types
        self.nlayers = 0

    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, edge_types, mask_neight, mask_time):
        use_cuda = next(self.parameters()).is_cuda
        batch_size, neigh_number, time_steps, input_dim = neighbors_input.size()

        neighbors_output = Variable(torch.zeros(batch_size, time_steps, self.hidden_dim * self.num_edge_types))
        if use_cuda:
            neighbors_output = neighbors_output.cuda()

        # node processing
        node_output = self.node_prj(node_input)

        # neighor processing
        edge_types = edge_types.unsqueeze(-1)
        neighbors_input = neighbors_input.unsqueeze(-2) * edge_types
        edge_mask = torch.sum(edge_types, dim=1)
        edge_mask[edge_mask > 1] = 1.
        neighbors_input = torch.sum(neighbors_input, dim=1)



        for t_idx in range(self.num_edge_types):
            output_group_by_type = self.neight_prj[t_idx](neighbors_input[:, :, t_idx])
            masked_output_group_by_type = output_group_by_type * edge_mask[:, :, t_idx]
            neighbors_output[:, :, t_idx * self.hidden_dim:(t_idx + 1) * self.hidden_dim] = masked_output_group_by_type

        output = self.prj(torch.cat((node_output, neighbors_output), dim=-1))
        return output


class NodeInterpolation(BaseNet):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=0.1):
        super(NodeInterpolation, self).__init__()
        self.node_prj = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                      nn.Dropout(dropout_prob),
                                      nn.ELU())

        # self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        self.prj = nn.Sequential(nn.Linear(hidden_dim, output_dim),
                                 nn.ELU())

        self.name = "NodeInterpolation"


        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nlayers = 0
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def forward(self, node_input, node_hidden, neighbors_input, neighbors_hidden, edge_types, mask_neight, mask_time):
        batch_size, neigh_number, time_steps, input_dim = neighbors_input.size()
        node_output = self.node_prj(node_input)
        output = self.prj(node_output)
        return output



