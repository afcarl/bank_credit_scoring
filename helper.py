from __future__ import unicode_literals, print_function, division
from io import open
import os
from torch.utils.data import Dataset
from collections import namedtuple
import torch
import pickle
import msgpack
import numpy as np

TIMESTAMP = ["2016-06-30", "2016-07-31", "2016-08-31", "2016-09-30", "2016-10-31", "2016-11-30", "2016-12-31",
             "2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30", "2017-05-31", "2017-06-30"]

REF_DATE = "2018-01-01"
DATE_FORMAT = "%Y-%m-%d"


use_cuda = torch.cuda.is_available()
TENSOR_TYPE = dict(f_tensor=torch.cuda.FloatTensor if use_cuda else torch.FloatTensor,
                   l_tensor=torch.cuda.LongTensor if use_cuda else torch.LongTensor,
                   i_tensor=torch.cuda.IntTensor if use_cuda else torch.IntTensor,
                   u_tensor=torch.cuda.ByteTensor if use_cuda else torch.ByteTensor)


def msg_unpack(file_name):
    with open(file_name, 'rb') as infile:
        data = msgpack.unpack(infile)
    return data

def msg_pack(data, file_name):
    with open(file_name, "wb") as outfile:
        msgpack.pack(data, outfile)


def get_param_numbers(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def hookFunc(module, gradInput, gradOutput):
    if np.isnan(gradInput[0].data.numpy()).any():
        print(gradInput[0])
    if np.isnan(gradOutput[0].data.numpy()).any():
        print(gradOutput[0])


def get_attn_mask(ngh_msk, time_window, size, use_cuda=False):
    time_mask = get_time_mask(time_window, size)
    mask = get_neigh_mask(ngh_msk, time_mask, size)

    if use_cuda:
        mask = mask.cuda()

    return mask

def get_neigh_mask(ngh_msk, time_mask, size):
    batch_size, neighbors, time_steps, hidden_dim = size
    for b_idx, num_ngh in enumerate(ngh_msk):
        time_mask[b_idx, :, num_ngh*time_steps:].fill_(1)

    return time_mask


def get_time_mask(time_size, size):
    ''' Get an attention mask to avoid using the subsequent info.'''
    batch_size, max_neighbors, time_steps, hidden_dim = size
    upper_mask = torch.from_numpy(np.triu(np.ones((batch_size, time_steps, time_steps)), k=1).astype('uint8'))
    lower_mask = torch.from_numpy(np.triu(np.ones((batch_size, time_steps, time_steps)), k=time_size).astype('uint8'))
    mask = upper_mask + lower_mask.transpose(1, 2)

    return mask.repeat(1, 1, max_neighbors + 1)


def get_temperature(max_temp, min_temp, decadicy_iteration, total_iterations=None):
    '''
    get the temperature mask
    :param max_temp: max temperature value
    :param low_temp: min temperature value
    :param decadicy_iteration: number of iteration with decadiment
    :param total_iterations: total number of iteration
    :return:
    '''
    if total_iterations == None:
        total_iterations = decadicy_iteration

    mask = np.linspace(max_temp, min_temp, decadicy_iteration).astype(np.float32)
    if decadicy_iteration < total_iterations:
        to_add = np.array([min_temp]*(total_iterations-decadicy_iteration)).astype(np.float32)
        mask = np.concatenate((mask, to_add), axis=0)
    return torch.from_numpy(mask)




def mse(input, target):
    return torch.mean((input - target) ** 2)

def rmse(input, target):
    return mse(input, target) ** 0.5

def accuracy(predict, target):
    correct = (target.eq(predict.round())).sum()
    return correct.float() / predict.size(0)


def get_embeddings(data_dir, prefix=""):
    use_cuda = torch.cuda.is_available()

    input_embeddings = pickle.load(open(os.path.join(data_dir, prefix + "input_embeddings.bin"), "rb"))
    target_embeddings = pickle.load(open(os.path.join(data_dir, prefix + "target_embeddings.bin"), "rb"))
    neighbor_embeddings = pickle.load(open(os.path.join(data_dir, prefix + "neighbor_embeddings.bin"), "rb"))
    seq_len = torch.LongTensor([4]*input_embeddings.size(0))

    if use_cuda:
        input_embeddings = input_embeddings.cuda()
        target_embeddings = target_embeddings.cuda()
        neighbor_embeddings = neighbor_embeddings.cuda()
        seq_len = seq_len.cuda()

    if target_embeddings.dim() == 2:
        target_embeddings = target_embeddings.unsqueeze(-1)

    return input_embeddings, target_embeddings, neighbor_embeddings, seq_len


def get_customer_embeddings(data_dir, prefix=""):
    """
    :param base_path: directory containing the data
    :param file_name: filename containing the customers data
    :param neighbors_file_name: file name containing the neighbor data
    :param embedding_dim: dimension of the input embedding
    :param input_ts_len: time length of the input sequence
    :param output_ts_len: time length of the target sequence
    :param risk_tsfm: transformer for the risk features
    :param attribute_tsfm: transformer for the attribute features
    :return:
    """
    use_cuda = torch.cuda.is_available()

    input_embeddings = torch.load(os.path.join(data_dir, "customers_embed.pt"))
    target_embeddings = torch.load(os.path.join(data_dir, "targets_embed.pt"))
    neighbor_embeddings = torch.load(os.path.join(data_dir, "neighbors_embed.pt"))
    ngh_msk = torch.load(os.path.join(data_dir, "ngh_msk.pt")).byte()

    num_customers = input_embeddings.size(0)

    if use_cuda:
        input_embeddings = input_embeddings.cuda()
        target_embeddings = target_embeddings.cuda()
        neighbor_embeddings = neighbor_embeddings.cuda()
        ngh_msk = ngh_msk.cuda()

    if target_embeddings.dim() == 2:
        target_embeddings = target_embeddings.unsqueeze(-1)

    assert num_customers == target_embeddings.size(0)
    assert num_customers == neighbor_embeddings.size(0)
    assert num_customers == ngh_msk.size(0)


    return input_embeddings, target_embeddings, neighbor_embeddings, ngh_msk



class CustomDataset(Dataset):
    def __init__(self, base_path, file_name):
        self.customers_list = torch.LongTensor(pickle.load(open(os.path.join(base_path, file_name), "rb")))

    def __len__(self):
        return len(self.customers_list)

    def __getitem__(self, idx):
        c_idx = self.customers_list[idx]
        return c_idx


class PositionwiseFeedForward(torch.nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Conv1d(d_hid, d_inner_hid, 1) # position-wise
        self.w_2 = torch.nn.Conv1d(d_inner_hid, d_hid, 1) # position-wise
        self.layer_norm = LayerNorm(d_hid)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.w_1(x.transpose(1, 2))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class LayerNorm(torch.nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        ret = ((self.gamma / (std + self.eps)) * (x - mean)) + self.beta
        return ret

class BiLinearProjection(torch.nn.Module):
    def __init__(self, in_out_dim, baias_size, transpose=True):
        super(BiLinearProjection, self).__init__()
        self.W = torch.nn.Parameter(TENSOR_TYPE["f_tensor"](in_out_dim, in_out_dim))
        self.transpose = transpose
        if baias_size > 0:
            self.b = torch.nn.Parameter(TENSOR_TYPE["f_tensor"](baias_size))

    def forward(self, node, neighbor):
        output = node.matmul(self.W)
        if self.transpose:
            output = output.matmul(neighbor.transpose(1, 2))
        else:
            output = output.matmul(neighbor)
        return output + self.b

class BaseNet(torch.nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.criterion = torch.nn.MSELoss()

    def reset_parameters(self):
        """
        reset the network parameters using xavier init
        :return:
        """
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                torch.nn.init.xavier_normal(p.data)

    def init_hidden(self, batch_size):
        """
        generate a new hidden state to avoid the back-propagation to the beginning to the dataset
        :return:
        """
        # weight = next(self.parameters()).data
        # hidden = torch.autograd.Variable(weight.new(self.nlayers, batch_size, self.hidden_dim).zero_())
        hidden = torch.autograd.Variable(torch.zeros((self.nlayers, batch_size, self.hidden_dim)))
        if torch.cuda.is_available():
            hidden = hidden.cuda()
        return hidden

        # return Variable(weight.new(batch_size, self.hidden_dim).zero_())

    def compute_loss(self, predict, target):
        return self.criterion(predict, target)

    def compute_error(self, predict, target):
        return rmse(predict, target)


def ensure_dir(file_path):
    '''
    Used to ensure the creation of a directory when needed
    :param file_path: path to the file that we want to create
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return file_path


class AttrProxy(object):
    """Translates index lookups into attribute lookups."""
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))