import pickle
import torch
import random
from helper import ensure_dir
from os import path


BASE_DIR = path.join("..", "..", "data", "sintetic")

def generate_embedding(dim, num_neighbors):
    input_embeddings = torch.FloatTensor(dim[0], dim[1], 1).normal_(5, 4).int().float()
    input_embeddings = torch.clamp(input_embeddings, 0, 10)
    neighbor_embeddings = torch.FloatTensor(dim[0], num_neighbors, dim[1], 1).zero_()
    target_embeddings = torch.FloatTensor(dim[0]).zero_()

    neighbors = {}
    for idx in range(input_embeddings.size(0)):
        neighbors[idx] = random.sample(range(input_embeddings.size(0)), num_neighbors)
        n_embedding = torch.stack([input_embeddings[n_idx] for n_idx in neighbors[idx]], dim=0)
        neighbor_embeddings[idx] = n_embedding
        target_embeddings[idx] = torch.mean(torch.cat((input_embeddings[idx, -1], n_embedding[:, -1].squeeze()), dim=0))
        # if input_embeddings[idx, -1, 0] >= 5:
        #     target_embeddings[idx] = torch.mean(n_embedding[int(num_neighbors/2):, -1])
        # else:
        #     target_embeddings[idx] = torch.mean(n_embedding[:int(num_neighbors/2), -1])

    return input_embeddings, target_embeddings, neighbor_embeddings


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
    input_embeddings, target_embeddings, neighbor_embeddings = generate_embedding((12000, 10), 4)

    pickle.dump(input_embeddings, open(ensure_dir(path.join(BASE_DIR, "simple_input_embeddings.bin")), "wb"))
    pickle.dump(target_embeddings, open(path.join(BASE_DIR, "simple_target_embeddings.bin"), "wb"))
    pickle.dump(neighbor_embeddings, open(path.join(BASE_DIR, "simple_neighbor_embeddings.bin"), "wb"))

    train_dataset, eval_dataset, test_dataset = split_training_test_dataset(list(range(input_embeddings.size(0))), e_t_size=1000)
    pickle.dump(train_dataset, open(path.join(BASE_DIR, "simple_train_dataset.bin"), "wb"))
    pickle.dump(eval_dataset, open(path.join(BASE_DIR, "simple_eval_dataset.bin"), "wb"))
    pickle.dump(test_dataset, open(path.join(BASE_DIR, "simple_test_dataset.bin"), "wb"))
