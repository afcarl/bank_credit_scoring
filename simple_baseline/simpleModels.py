import torch
from helper import rmse, TIMESTAMP
import pickle
from os.path import join as path_join

import torch.nn as nn


class LinearCombinationMean(nn.Module):
    """compute the simple mean of the previous timestemp and perform a linear combination between customers and neighbors"""
    def __init__(self, features_size, output_size=1):
        super(LinearCombinationMean, self).__init__()
        self.linear = nn.Linear(features_size, output_size)
        self.criterion = nn.MSELoss()

        self.reset_parameters()

    def forward(self, c_input, n_input, s_len):
        c_mean = torch.mean(c_input[:, :, 2], dim=1)
        n_mean = torch.mean(n_input[:, :, :, 2].sum(dim=1).div(s_len.unsqueeze(-1).expand(-1, len(TIMESTAMP)-1).float()), dim=1)
        n_mean[n_mean != n_mean] = 0
        return self.linear(torch.stack((c_mean, n_mean), dim=1))

    def compute_loss(self, predict, target):
        return self.criterion(predict, target)

    def compute_error(self, predict, target):
        return rmse(predict, target)

    def reset_parameters(self):
        """
        reset the network parameters using xavier init
        :return:
        """
        for p in self.parameters():
            if len(p.data.shape) == 1:
                p.data.fill_(0)
            else:
                nn.init.xavier_uniform(p.data)

class SimpleMean(object):
    """compute the simple mean of the previous timestemp"""
    def __init__(self):
        pass

    def forward(self, input):
        return torch.mean(input[:, :, 2], dim=1)

    def compute_error(self, predict, target):
        return rmse(predict, target)
