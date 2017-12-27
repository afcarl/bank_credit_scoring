import torch
from helper import rmse
class SimpleMean(object):
    """compute the simple mean of the previous timestemp"""
    def __init__(self):
        pass

    def forward(self, input):
        return torch.mean(input[:, :, 2], dim=1)

    def compute_error(self, predict, target):
        return rmse(predict, target)
