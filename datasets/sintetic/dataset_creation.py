import pickle
import torch
import random
from helper import ensure_dir
from os import path
from math import pi as PI
import numpy as np
BASE_DIR = path.join("..", "..", "data", "sintetic")



def triangle2(length, amplitude=6):
    section = length // 4
    x = np.linspace(0, amplitude, section+1)
    mx = -x
    return np.r_[x, x[-2::-1], mx[1:], mx[-2:0:-1]]


def generate_to_test(dim, num_neighbors):
    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).uniform_(0, 10)
    input_embeddings = torch.clamp(input_embeddings, 0, 9)
    input_embeddings, _ = torch.sort(input_embeddings, 1)
    neighbor_embeddings, _ = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).uniform_(0, 10).clamp(0, 9).sort(dim=1)
    target_embeddings = torch.FloatTensor(dim[0], dim[1]).zero_()

    return input_embeddings, target_embeddings, neighbor_embeddings, "test"

def generate_triangular_embedding(dim, num_neighbors):
    amplitudes = torch.Tensor(dim[0]).uniform_(5, 10).int().float()
    split_points = torch.Tensor(dim[0]).uniform_(0, 2).int()
    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).zero_()
    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], dim[1], 1).zero_()

    for idx in range(dim[0]):
        input_embedding = triangle2(2*dim[1], amplitude=amplitudes[idx])
        input_embedding += np.random.randint(-2, 2, 1)
        input_embedding = input_embedding.astype(np.int32)
        split_point = split_points[idx]
        neighbors_split_offset = split_point + np.random.randint(1, 3)
        neighbors_multiply_offset = np.random.randint(1, 3, num_neighbors)
        input_embeddings[idx] = torch.from_numpy(input_embedding[split_point:split_point+dim[1]])
        n_embedding = np.matmul(np.expand_dims(input_embedding, axis=-1), np.expand_dims(neighbors_multiply_offset, axis=0)).T
        for n_idx in range(num_neighbors):
            neighbor_embeddings[idx, n_idx] = torch.from_numpy(n_embedding[n_idx, neighbors_split_offset:neighbors_split_offset+10].astype(np.int32)).float()
        target_embeddings[idx] = torch.from_numpy(input_embedding[split_point+1:split_point+dim[1]+1])
    return input_embeddings, target_embeddings, neighbor_embeddings, "tr"


def generate_noise_embedding(dim, num_neighbors, offset_sampler = torch.distributions.Categorical(torch.Tensor([ 0.25, 0.25, 0.25]))):
    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).uniform_(0, 10).int().float()
    input_embeddings, _ = torch.sort(input_embeddings, 1)
    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], dim[1]).zero_()

    for idx in range(input_embeddings.size(0)):
        n_embedding = torch.FloatTensor(num_neighbors, dim[1], 1).zero_()
        n_embedding[0] = input_embeddings[idx]
        n_embedding[-1] = torch.FloatTensor(dim[1], 1).uniform_(0, 10)
        n_embedding[1:-1] = input_embeddings[idx].repeat(2, 1, 1) + offset_sampler.sample(sample_shape=(2, 1, 1)).float()
        neighbor_embeddings[idx] = n_embedding.int().float()
        target_embeddings[idx] = torch.sum(torch.cat((input_embeddings[idx].unsqueeze(0), n_embedding[:-1]), dim=0), dim=0)
        # if input_embeddings[idx, -1, 0] >= 5:
        #     target_embeddings[idx] = torch.mean(n_embedding[int(num_neighbors/2):, -1])
        # else:
        #     target_embeddings[idx] = torch.mean(n_embedding[:int(num_neighbors/2), -1])

    return input_embeddings, target_embeddings, neighbor_embeddings, "simple"


def generate_middle_select_embedding(dim, num_neighbors, pos=5):
    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).uniform_(0, 10)

    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], 1).zero_()

    for idx in range(dim[0]):
        n_embedding = torch.FloatTensor(num_neighbors, dim[1], 1).uniform_(0, 10).int().float()
        neighbor_embeddings[idx] = n_embedding
        target_embeddings[idx] = torch.sum(torch.cat((input_embeddings[idx, pos].unsqueeze(0), n_embedding[:, pos]), dim=0), dim=0)

    return input_embeddings, target_embeddings, neighbor_embeddings, "ms"


def generate_timeshifted_embedding(dim, num_neighbors):
    time_idx = np.sort(np.random.uniform(low=0, high=10, size=(dim[0], dim[1])).astype(np.int32).clip(0,9), axis=1)
    idx = np.array(range(dim[1])).astype(np.int32)
    while (time_idx[:, idx] > idx).any():
        time_idx[time_idx[:, idx] > idx] -= 1
    time_idx = torch.from_numpy(time_idx).long()

    input_embeddings = time_idx.float().unsqueeze(-1)
    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], dim[1]).zero_()

    for idx in range(dim[0]):
        n_embedding,_ = torch.FloatTensor(num_neighbors, dim[1], 1).uniform_(0, 10).int().float().sort(dim=1)
        neighbor_embeddings[idx] = n_embedding
        target_embeddings[idx] = torch.sum(torch.cat((input_embeddings[idx].unsqueeze(0), n_embedding[:, time_idx[idx]]), dim=0), dim=0)

    return input_embeddings, target_embeddings, neighbor_embeddings, "ts"

def generate_sinusoidal_embedding(dim, num_neighbors, offset_sampler=torch.distributions.Categorical(torch.Tensor([0.25, 0.25, 0.25]))):

    x, _ = torch.FloatTensor(dim[0], 1).uniform_(0,3).sort(dim=0)
    x = x.expand(-1, dim[1])
    x = x + torch.FloatTensor([(PI/10) * i for i in range(10)])



    input_embeddings = torch.sin(x).unsqueeze(-1)
    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], dim[1]).zero_()

    for idx in range(dim[0]):
        n_embedding = torch.FloatTensor(num_neighbors, dim[1], 1).zero_()
        n_embedding[0] = input_embeddings[idx]
        n_embedding[1] = torch.sin(x[idx] - offset_sampler.sample(sample_shape=(1, 1, 1)).float())
        n_embedding[2] = torch.sin(x[idx] - offset_sampler.sample(sample_shape=(1, 1, 1)).float())
        n_embedding[3] = torch.sin(x[idx] + offset_sampler.sample(sample_shape=(1, 1, 1)).float())

        neighbor_embeddings[idx] = n_embedding
        target_embeddings[idx] = torch.mean(torch.cat((input_embeddings[idx].unsqueeze(0), n_embedding), dim=0), dim=0)

    return input_embeddings, target_embeddings, neighbor_embeddings, "sin"

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
    input_embeddings, target_embeddings, neighbor_embeddings, prefix = generate_triangular_embedding((10000, 10), 4)

    pickle.dump(input_embeddings, open(ensure_dir(path.join(BASE_DIR, prefix+"_input_embeddings.bin")), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, prefix+"_target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, prefix+"_neighbor_embeddings.bin"), "wb"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(list(range(input_embeddings.size(0))), e_t_size=1000)
    pickle.dump(train_dataset, open(path.join(BASE_DIR, prefix+"_train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, prefix+"_eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, prefix+"_test_dataset.bin"), "wb"))
