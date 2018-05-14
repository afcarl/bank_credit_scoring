import pickle
from os import path
from datasets.utils import BASE_DIR, MyBidict, SiteInfo, one_hot_conversion
import pandas as pd
from collections import OrderedDict
from bidict import bidict
import torch
from helper import ensure_dir
import re
import geopy.distance
from numpy import array, arange, exp, log, nan




if __name__ == "__main__":
    input_embeddings = torch.load(path.join(BASE_DIR, "utility", "input_embeddings.pt"))
    target_embeddings = torch.load(path.join(BASE_DIR, "utility", "target_embeddings.pt"))
    neighbor_embeddings = torch.load(path.join(BASE_DIR, "utility", "neighbor_embeddings.pt"))
    site_to_idx = torch.load(path.join(BASE_DIR, "utility", "site_to_idx.pt"))
    site_to_exp_idx = torch.load(path.join(BASE_DIR, "utility", "site_to_exp_idx.pt"))
    print(site_to_idx[6])

    print(input_embeddings[site_to_exp_idx[6]])
    print(target_embeddings[site_to_exp_idx[6]])
    print(neighbor_embeddings[site_to_exp_idx[6]])





