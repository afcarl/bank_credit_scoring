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
    split_points = torch.Tensor(dim[0]).uniform_(1, 5).int()
    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).zero_()
    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], dim[1], 1).zero_()

    edge_type = torch.ones(dim[0], num_neighbors, dim[1], 1)
    neigh_mask = torch.zeros(dim[0], num_neighbors)


    for idx in range(dim[0]):
        input_embedding = triangle2(2*dim[1], amplitude=amplitudes[idx])
        input_embedding += np.random.randint(-2, 2, 1)
        input_embedding = input_embedding.astype(np.int32)
        split_point = split_points[idx]
        neighbors_split_offset = split_point + np.random.randint(1, 3)
        neighbors_multiply_offset = np.random.randint(1, 3, num_neighbors)
        input_embeddings[idx] = torch.from_numpy(input_embedding[split_point:split_point+dim[1]]).unsqueeze(1)
        n_embedding = np.matmul(np.expand_dims(input_embedding, axis=-1), np.expand_dims(neighbors_multiply_offset, axis=0)).T
        for n_idx in range(num_neighbors):
            neighbor_embeddings[idx, n_idx] = torch.from_numpy(n_embedding[n_idx,
                                                               neighbors_split_offset:neighbors_split_offset+10]
                                                               .astype(np.int32)).float().unsqueeze(1)

        target_embeddings[idx] = torch.from_numpy(input_embedding[split_point+1:split_point+dim[1]+1]).unsqueeze(1)
    return input_embeddings, target_embeddings, neighbor_embeddings, edge_type, neigh_mask.byte(), "tr_neigh-{}".format(num_neighbors)

def generate_noise_triangular_embedding(dim, max_num_neighbors, max_relevant_neighbours,):

    prefix = "noise_tr"
    prefix += "_neigh-{}_rel-{}".format(max_num_neighbors, max_relevant_neighbours)

    amplitudes = torch.Tensor(dim[0]).uniform_(5, 10).int().float()
    split_points = torch.Tensor(dim[0]).uniform_(0, 2).int()
    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).zero_()
    neighbor_embeddings = torch.FloatTensor(dim[0], max_num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0], dim[1], 1).zero_()

    edge_type = torch.ones(dim[0], max_num_neighbors, dim[1], 1)
    neigh_mask = torch.zeros(dim[0], max_num_neighbors)

    non_relevant = max_num_neighbors - max_relevant_neighbours
    for idx in range(dim[0]):
        input_embedding = triangle2(2*dim[1], amplitude=amplitudes[idx])
        input_embedding += np.random.randint(-2, 2, 1)
        input_embedding = input_embedding.astype(np.int32)
        split_point = split_points[idx]
        neighbors_split_offset = split_point + np.random.randint(1, 3)
        neighbors_multiply_offset = np.random.randint(1, 3, max_num_neighbors)
        input_embeddings[idx] = torch.from_numpy(input_embedding[split_point:split_point+dim[1]]).unsqueeze(1)
        n_embedding = np.matmul(np.expand_dims(input_embedding, axis=-1), np.expand_dims(neighbors_multiply_offset, axis=0)).T

        for n_idx in range(max_relevant_neighbours):
            neighbor_embeddings[idx, n_idx] = torch.from_numpy(n_embedding[n_idx, neighbors_split_offset:neighbors_split_offset+10].astype(np.int32)).float().unsqueeze(1)

        neighbor_embeddings[idx, max_relevant_neighbours:] = torch.FloatTensor(non_relevant, dim[1], 1).uniform_(-10, 10).long().float()
        target_embeddings[idx] = torch.from_numpy(input_embedding[split_point+1:split_point+dim[1]+1]).unsqueeze(1)

    return input_embeddings, target_embeddings, neighbor_embeddings, edge_type, neigh_mask.byte(), prefix



def generate_noise_embedding(dim,
                             max_num_neighbors,
                             max_relevant_neighbours,
                             offset_sampler=torch.distributions.Categorical(torch.Tensor([0.2, 0.2, 0.2, 0.2, 0.2])),
                             randomize_neighbors=False,
                             dynamic_num_neighbors=False):
    prefix = "simple"
    if randomize_neighbors:
        prefix += "_random"
    if dynamic_num_neighbors:
        prefix += "_dynamic"

    prefix += "_neigh-{}_rel-{}".format(max_num_neighbors, max_relevant_neighbours)

    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).uniform_(1, 10).int().float()
    input_embeddings, _ = torch.sort(input_embeddings, 1)
    neighbor_embeddings = torch.zeros(dim[0], max_num_neighbors, dim[1], 1)
    target_embeddings = torch.zeros(dim[0], dim[1], 1)

    edge_type = torch.ones(dim[0], max_num_neighbors, dim[1], 1)
    neigh_mask = torch.zeros(dim[0], max_num_neighbors)



    for idx in range(input_embeddings.size(0)):
        n_embedding = torch.zeros(max_num_neighbors, dim[1], 1)

        if dynamic_num_neighbors:
            num_neighbors = np.random.randint(2, max_num_neighbors+1)
            relevant_neighbours = num_neighbors - 1
        else:
            num_neighbors = max_num_neighbors
            relevant_neighbours = max_relevant_neighbours

        assert num_neighbors - 1 >= relevant_neighbours, "invalid number of neighbours"
        non_relevant_neighbours = num_neighbors - relevant_neighbours
        assert non_relevant_neighbours >= 0, "invalid number of non-relevant neighbours"
        assert non_relevant_neighbours + relevant_neighbours == num_neighbors


        # n_embedding[0] = input_embeddings[idx]
        n_embedding[relevant_neighbours:relevant_neighbours+1] = torch.FloatTensor(non_relevant_neighbours, dim[1], 1).uniform_(-5, 5)
        n_embedding[:relevant_neighbours] = input_embeddings[idx].repeat(relevant_neighbours, 1, 1) + \
                                               offset_sampler.sample(sample_shape=(relevant_neighbours, 1, 1)).float()
        # compute target value
        n_embedding = n_embedding.int().float()
        target_embeddings[idx] = torch.mean(torch.cat((input_embeddings[idx].unsqueeze(0), n_embedding[:relevant_neighbours]), dim=0), dim=0)

        # compute mask
        if num_neighbors < max_num_neighbors:
            neigh_mask[idx, num_neighbors:] = 1

        neighbor_embeddings[idx] = n_embedding

    return input_embeddings, target_embeddings, neighbor_embeddings, edge_type, neigh_mask.byte(), prefix


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
    input_embeddings, target_embeddings, neighbor_embeddings, edge_type, mask_neigh, prefix = generate_noise_triangular_embedding((12000, 10), 1000, 4)

    torch.save(input_embeddings, ensure_dir(path.join(BASE_DIR, prefix+"_input_embeddings.pt")))
    torch.save(target_embeddings, ensure_dir(path.join(BASE_DIR, prefix + "_target_embeddings.pt")))
    torch.save(neighbor_embeddings, ensure_dir(path.join(BASE_DIR, prefix + "_neighbor_embeddings.pt")))
    torch.save(edge_type, ensure_dir(path.join(BASE_DIR, prefix + "_edge_type.pt")))
    torch.save(mask_neigh, ensure_dir(path.join(BASE_DIR, prefix + "_mask_neighbor.pt")))


    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(list(range(input_embeddings.size(0))), e_t_size=1000)
    torch.save(train_dataset, path.join(BASE_DIR, prefix+"_train_dataset.pt"))
    torch.save(eval_dataset, path.join(BASE_DIR, prefix+"_eval_dataset.pt"))
    torch.save(test_dataset, path.join(BASE_DIR, prefix+"_test_dataset.pt"))