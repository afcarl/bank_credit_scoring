import pickle
from os import path
import torch

def get_sintetic_embeddings(data_dir, prefix=""):
    input_embeddings = pickle.load(open(path.join(data_dir, prefix + "input_embeddings.bin"), "rb"))
    target_embeddings = pickle.load(open(path.join(data_dir, prefix + "target_embeddings.bin"), "rb"))
    neighbor_embeddings = pickle.load(open(path.join(data_dir, prefix + "neighbor_embeddings.bin"), "rb"))
    seq_len = torch.LongTensor([4]*input_embeddings.size(0))
    return input_embeddings, target_embeddings, neighbor_embeddings, seq_len