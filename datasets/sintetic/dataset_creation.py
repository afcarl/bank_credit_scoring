import pickle
import torch
import random
from helper import ensure_dir
from os import path
from math import pi as PI
import numpy as np
BASE_DIR = path.join("..", "..", "data", "sintetic")

def generate_simple_embedding(dim, num_neighbors, offset_sampler= torch.distributions.Categorical(torch.Tensor([ 0.25, 0.25, 0.25]))):
    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).uniform_(0, 2)
    input_embeddings = torch.clamp(input_embeddings, 0, 4)
    input_embeddings, _ = torch.sort(input_embeddings, 1)
    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], dim[1]).zero_()

    for idx in range(input_embeddings.size(0)):
        n_embedding = torch.FloatTensor(num_neighbors, dim[1], 1).zero_()
        n_embedding[0] = input_embeddings[idx]
        n_embedding[-1] = torch.FloatTensor(dim[1], 1).uniform_(5, 15)
        n_embedding[1:-1] = input_embeddings[idx].repeat(2, 1, 1) + offset_sampler.sample(sample_shape=(2, 1, 1)).float()
        neighbor_embeddings[idx] = n_embedding
        target_embeddings[idx] = torch.mean(torch.cat((input_embeddings[idx].unsqueeze(0), n_embedding[:-1]), dim=0), dim=0)
        # if input_embeddings[idx, -1, 0] >= 5:
        #     target_embeddings[idx] = torch.mean(n_embedding[int(num_neighbors/2):, -1])
        # else:
        #     target_embeddings[idx] = torch.mean(n_embedding[:int(num_neighbors/2), -1])

    return input_embeddings, target_embeddings, neighbor_embeddings


def generate_sinusoidal_embedding(dim, num_neighbors, offset_sampler= torch.distributions.Categorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))):
    x = np.sort(np.random.uniform(low=0, high=10, size=(dim[0], dim[1])).astype(np.int32).clip(0,9), axis=1)
    idx = np.array(range(dim[1])).astype(np.int32)
    while (x[:, idx] > idx).any():
        x[x[:, idx] > idx] -= 1

    input_embeddings = torch.from_numpy(x).long()
    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], dim[1]).zero_()

    for idx in range(input_embeddings.size(0)):
        n_embedding = torch.FloatTensor(num_neighbors, dim[1], 1).zero_()
        x, _ = torch.FloatTensor(dim[1]).uniform_(-2*PI, 2*PI).sort()
        n_embedding[0] = torch.sin(x)
        n_embedding[1] = torch.cos(x)
        n_embedding[2] = torch.sin(x*2)
        n_embedding[3] = torch.FloatTensor(dim[1], 1).uniform_(0, 10)
        neighbor_embeddings[idx] = n_embedding
        target_embeddings[idx] = torch.mean(n_embedding[:, input_embeddings[idx]][:-1], dim=0)

    return input_embeddings.float().unsqueeze(-1), target_embeddings, neighbor_embeddings

def split_training_test_dataset(_ids, e_t_size=25000):
    """
    split the dataset in training/testing/eval dataset
    :param stock_ids: id of each stock
    :return:
    """

    test_dataset = random.sample(_ids, e_t_size)
    for s_idx in test_dataset:
        _ids.remove(s_idx)

    eval_dataset = random.sample(_ids, e_t_size)
    for s_idx in eval_dataset:
        _ids.remove(s_idx)


    return _ids, eval_dataset, test_dataset





if __name__ == "__main__":
    input_embeddings, target_embeddings, neighbor_embeddings = generate_sinusoidal_embedding((10000, 10), 4)

    pickle.dump(input_embeddings, open(ensure_dir(path.join(BASE_DIR, "sin_input_embeddings.bin")), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, "sin_target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, "sin_neighbor_embeddings.bin"), "wb"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(list(range(input_embeddings.size(0))), e_t_size=1000)
    pickle.dump(train_dataset, open(path.join(BASE_DIR, "sin_train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "sin_eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "sin_test_dataset.bin"), "wb"))
