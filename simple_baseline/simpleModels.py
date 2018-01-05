import torch
from helper import rmse, TIMESTAMP
from numpy import convolve
import pickle
from os.path import join as path_join

import torch.nn as nn


class LinearCombinationMean(nn.Module):
    """compute the simple mean of the previous timestemp and perform a linear combination between customers and neighbors"""
    def __init__(self, mean_len, features_size, output_size=1):
        super(LinearCombinationMean, self).__init__()
        self.mean_len = mean_len
        self.linear = nn.Linear(features_size, output_size)
        self.criterion = nn.MSELoss()

        self.reset_parameters()

    def forward(self, c_input, n_input, s_len):
        c_mean = torch.mean(c_input[:, :self.mean_len, 2], dim=1)
        n_mean = torch.mean(n_input[:, :, :self.mean_len, 2].sum(dim=1).div(s_len.unsqueeze(-1).expand(-1, self.mean_len).float()), dim=1)
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


class SimpleFeatureMean(nn.Module):
    """compute the linear combination between simple mean and the output feature"""
    def __init__(self, linear_input_size, linear_output_size, input_seq_len, output_seq_len, feature_len):
        super(SimpleFeatureMean, self).__init__()
        mean_feature_len = input_seq_len - output_seq_len + 1
        assert (input_seq_len - mean_feature_len) + 1 > feature_len, "rolling mean is too long. Not enough point to compute the rolling mean error"
        assert mean_feature_len >= feature_len, "feature vector longer than mean rolling mean. Impossible to compute the back error"
        self.rolling_mean_len = (input_seq_len - mean_feature_len) + 1
        self.mean_len = mean_feature_len
        self.feature_len = feature_len

        self.mean_weight = torch.autograd.Variable(torch.FloatTensor(1, 1, self.mean_len).fill_(1) / self.mean_len,
                                                   volatile=True)
        self.n_mean_weight = self.mean_weight.expand(44, 44, -1)
        self.linear = nn.Linear(linear_input_size, linear_output_size)
        self.criterion = nn.MSELoss()

        self.reset_parameters()

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

    def compute_loss(self, predict, target):
        return self.criterion(predict, target)

    def compute_error(self, predict, target):
        return rmse(predict, target)

    def neighborMeanForward(self, c_input, n_input, s_len):
        """
        1)compute the rolling mean for mean_len timestemps.
        3)compute the difference between inputs and outputs
        4)compute the difference between outputs and outputs


        2)compute the neighbors rolling mean for mean_len timestemps.
        5)compute the neighbor difference between inputs and outputs
        6)compute the neighbor difference between outputs and outputs

        :param c_input: customer input
        :param n_input: neighbors input
        :param s_len: sequence length (number of neighbors)
        :return:
        """
        f_c_ret = self.simpleMeanForward(c_input)
        f_n_ret = self.simpleMeanForward(n_input)

        f_n_ret[f_n_ret != f_n_ret] = 0 # set nan to 0
        f_n_ret_sum = f_n_ret.sum(dim=1) # sum over the neighbor to compute the mean

        n_mean = f_n_ret_sum.div(s_len.unsqueeze(-1).unsqueeze(-1).expand(f_n_ret_sum.size()).float())   # divide by the number of neighbors
        n_mean[n_mean != n_mean] = 0

        f_ret = torch.cat((f_c_ret, n_mean), dim=-1)
        return f_ret

        # n_running_mean, input = torch.transpose(running_mean, 1, 2), input.data.squeeze()  # remove channel dim
        # input_idx = torch.LongTensor(
        #     [range(i + self.mean_len, i + self.mean_len - self.feature_len, -1) for i in range(-1, running_mean.size(1) - 1)])


    def simpleMeanForward(self, input):
        """
        1)compute the rolling mean for mean_len timestemps.
        2)compute the difference between inputs and outputs
        """
        if input.dim() == 2:
            input = input.unsqueeze(1) # add input channel dimension
        running_mean = torch.nn.functional.avg_pool1d(input, kernel_size=self.mean_len, stride=1)  # remove variable(no need to learn filters weight)
        input_idx = torch.LongTensor(
            [list(range(self.rolling_mean_len + self.feature_len + i, self.rolling_mean_len + i - 1, -1)) for i in
             range(self.rolling_mean_len)])



        running_mean_idx = torch.LongTensor(
            [list(range(self.feature_len + i, i, -1)) for i in range(-1, running_mean.size(-1) - self.feature_len - 1)]) # compute time diff index
        slide_running_mean = running_mean[:, :, running_mean_idx]   # extract the previous value
        slide_running_mean = torch.cat((
            torch.autograd.Variable(
                torch.FloatTensor(slide_running_mean.size(0), slide_running_mean.size(1), self.feature_len,
                                  slide_running_mean.size(-1)).zero_()),
            slide_running_mean), dim=-2)  # padding 0
        running_mean = running_mean.unsqueeze(-1)

        # input_features = (running_mean - input[:, :, input_idx]) / running_mean
        output_features = (running_mean - slide_running_mean) / running_mean


        # f_ret = torch.cat((running_mean, input_features), dim=-1)
        f_ret = torch.cat((running_mean, output_features), dim=-1)
        # f_ret = torch.cat((running_mean, output_features, input_features), dim=-1)
        return f_ret.squeeze()


    def forward(self, input):
        output = torch.stack([self.linear(input[:, timestemp_idx, :]) for timestemp_idx in range(input.size(1))], dim=1)
        return output