from __future__ import unicode_literals, print_function, division
from io import open
import os
from torch.utils.data import Dataset
from collections import namedtuple
import torch
import pickle
import numpy as np
TIMESTAMP = ["2016-06-30", "2016-07-31", "2016-08-31", "2016-09-30", "2016-10-31", "2016-11-30", "2016-12-31",
             "2017-01-31", "2017-02-28", "2017-03-31", "2017-04-30", "2017-05-31", "2017-06-30"]

RISK_ATTRIBUTE = ["segmento",
                  "class_scoring_risk", "val_scoring_risk",
                  "class_scoring_ai", "val_scoring_ai",
                  "class_scoring_bi", "val_scoring_bi",
                  "class_scoring_cr", "val_scoring_cr",
                  "class_scoring_sd", "val_scoring_sd",
                  "pre_notching", "class_scoring_pre"]

C_ATTRIBUTE = ["ateco", "b_partner", "birth_date", "cod_uo", "country_code", "customer_kind", "customer_type", "region",
               "sae", "uncollectable_status", "zipcode"]

REF_DATE = "2018-01-01"
DATE_FORMAT = "%Y-%m-%d"

T_risk = namedtuple("T_risk", RISK_ATTRIBUTE)
T_attribute = namedtuple("T_attribute", C_ATTRIBUTE)
CustomerSample = namedtuple('CustomerSample', ['customer_id', 'risk', 'attribute'])
PackedNeighbor = namedtuple('PackedNeighbor', ['neighbors', 'seq_len'])
PackedWeight = namedtuple('PackedWeight', ['net_weight', 'time_weight'])

use_cuda = torch.cuda.is_available()
TENSOR_TYPE = dict(f_tensor=torch.cuda.FloatTensor if use_cuda else torch.FloatTensor,
                   l_tensor=torch.cuda.LongTensor if use_cuda else torch.LongTensor,
                   i_tensor=torch.cuda.IntTensor if use_cuda else torch.IntTensor,
                   u_tensor=torch.cuda.ByteTensor if use_cuda else torch.ByteTensor)



def get_attn_mask(size, use_cuda, time_size=10):
    ''' Get an attention mask to avoid using the subsequent info.'''
    batch_size, neighbors, time_steps, hidden_dim = size
    upper_mask = torch.from_numpy(np.triu(np.ones((batch_size, time_steps, time_steps)), k=1).astype('uint8'))
    lower_mask = torch.from_numpy(np.triu(np.ones((batch_size, time_steps, time_steps)), k=time_size).astype('uint8'))
    mask = upper_mask + lower_mask.transpose(1, 2)
    if use_cuda:
        mask = mask.cuda()
    return mask


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



def update_or_plot(i_iter):
    if i_iter == 0:
        return None
    else:
        return "append"


def get_max_length(path):
    return len(pickle.load(open(path, "rb")))



class RiskToTensor(object):
    def __init__(self, base_path):
        self.max_segmento = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("segmento")))
        self.transformer = self.__risk_tensor__

    def __call__(self, sample):
        return self.transformer(sample)

    def __risk_tensor__(self, risk):
        return torch.FloatTensor(risk)

    def __risk_tensor_one_hot__(self, risk):
        segmento_one_hot = torch.zeros((len(risk), self.max_segmento))
        segmento_idx = [timestemp.segmento for timestemp in risk]
        segmento_one_hot[:, segmento_idx] = 1

        return torch.cat((segmento_one_hot,
                          torch.FloatTensor(list(risk))[:, 1:]), dim=1)

class AttributeToTensor(object):
    def __init__(self, base_path):
        self.max_ateco = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("ateco")))
        self.max_b_partner = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("b_partner")))
        self.max_c_kind = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("c_kind")))
        self.max_c_type = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("c_type")))
        self.max_cod_uo = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("cod_uo")))
        self.max_country_code = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("country_code")))
        self.max_region = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("region")))
        self.max_sae = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("sae")))
        self.max_un_status = get_max_length(
            os.path.join(base_path, "dicts", "{}_dict.bin".format("uncollectable_status")))
        self.max_zipcode = get_max_length(os.path.join(base_path, "dicts", "{}_dict.bin".format("zipcode")))

        self.transformer = self.__attribute_tensor__

    def __call__(self, sample):
        return self.transformer(sample)

    def __attribute_tensor__(self, attribute):
        return torch.FloatTensor([attribute.ateco, attribute.birth_date, attribute.zipcode, attribute.b_partner,
                                  attribute.customer_kind, attribute.customer_type, attribute.cod_uo,
                                  attribute.country_code, attribute.region, attribute.sae, attribute.uncollectable_status])

    def __attribute_one_hot_tensor__(self, attribute):

        b_partner_one_hot = torch.zeros((self.max_b_partner))
        b_partner_one_hot[attribute.b_partner] = 1

        c_kind_one_hot = torch.zeros((self.max_c_kind))
        c_kind_one_hot[attribute.customer_kind] = 1

        c_type_one_hot = torch.zeros((self.max_c_type))
        c_type_one_hot[attribute.customer_type] = 1

        cod_uo_one_hot = torch.zeros((self.max_cod_uo))
        cod_uo_one_hot[attribute.cod_uo] = 1

        country_code_one_hot = torch.zeros((self.max_country_code))
        country_code_one_hot[attribute.country_code] = 1

        region_one_hot = torch.zeros((self.max_region))
        region_one_hot[attribute.region] = 1

        sae_one_hot = torch.zeros((self.max_sae))
        sae_one_hot[attribute.sae] = 1

        un_status_one_hot = torch.zeros((self.max_un_status))
        un_status_one_hot[attribute.uncollectable_status] = 1

        return torch.cat((torch.FloatTensor([attribute.ateco, attribute.birth_date, attribute.zipcode]),
                          b_partner_one_hot,
                          c_kind_one_hot,
                          c_type_one_hot,
                          cod_uo_one_hot,
                          country_code_one_hot,
                          region_one_hot,
                          sae_one_hot,
                          un_status_one_hot
                          ))

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


def get_customer_embeddings(base_path, file_name, neighbors_file_name, embedding_dim, input_ts_len, output_ts_len, risk_tsfm, attribute_tsfm):
    """
    generate the embedding.
    1) Read the raw features
    2) Convert the raw features in a vector format
    3) Write the features of each customer in the appropriate embedding matrix position
    4) Extract the target values
    5) Extract the neighbors values

    :param base_path: directory containing the data
    :param file_name: filename containing the customer data
    :param neighbors_file_name: file name containing the neighbor data
    :param embedding_dim: dimension of the input embedding
    :param input_ts_len: time length of the input sequence
    :param output_ts_len: time length of the target sequence
    :param risk_tsfm: transformer for the risk features
    :param attribute_tsfm: transformer for the attribute features
    :return:
    """
    customer_embeddings = pickle.load(open(os.path.join(base_path, file_name), "rb"))
    customer_neighbor_embeddings = pickle.load(open(os.path.join(base_path, neighbors_file_name), "rb"))

    assert input_ts_len < len(TIMESTAMP), "input seq_len bigger than total available timestemps"
    assert output_ts_len < len(TIMESTAMP), "output seq_len bigger than total available timestemps"

    cs_len = len(customer_embeddings) + 1 # number of contomers + 1 (null customers => idx = 0)

    input_embedding = torch.FloatTensor(cs_len, input_ts_len, embedding_dim).zero_()
    target_embedding = torch.FloatTensor(cs_len, output_ts_len).zero_()
    neighbor_embedding = torch.FloatTensor(cs_len, len(customer_neighbor_embeddings[1].neighbors),
                                           input_ts_len, embedding_dim).zero_()
    seq_len = torch.LongTensor(cs_len).zero_()

    for c_idx, attributes in customer_embeddings.items():
        c_id, c_risk, c_attribute = attributes

        torch_risk = risk_tsfm(c_risk)
        torch_attribute = attribute_tsfm(c_attribute)

        input_embedding[c_idx] = torch.cat((torch_risk[:input_ts_len, :], torch_attribute.expand(input_ts_len, torch_attribute.size(0))), dim=1)
        target_embedding[c_idx] = torch_risk[-output_ts_len:, 2]

    for c_idx in customer_embeddings.keys():
        n_embedding = input_embedding[customer_neighbor_embeddings[c_idx].neighbors]
        neighbor_embedding[c_idx] = n_embedding
        seq_len[c_idx] = customer_neighbor_embeddings[c_idx].seq_len

    return input_embedding, target_embedding, neighbor_embedding, seq_len



class CustomDataset(Dataset):
    def __init__(self, base_path, file_name):
        self.customers_list = torch.LongTensor(pickle.load(open(os.path.join(base_path, file_name), "rb")))

    def __len__(self):
        return len(self.customers_list)

    def __getitem__(self, idx):
        c_idx = self.customers_list[idx]
        return c_idx

class LayerNormalization(torch.nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = torch.nn.Parameter(torch.zeros(d_hid), requires_grad=True)
        self.b_2 = torch.nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

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

    def repackage_hidden_state(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == torch.autograd.Variable:
            return torch.autograd.Variable(h.data)
        else:
            return tuple(self.repackage_hidden_state(v) for v in h)

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