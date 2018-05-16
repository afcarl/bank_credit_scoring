import torch
import pickle
import random
from helper import get_embeddings
from os import path
BASE_DIR = path.join("..", "..", "data", "pems")

if __name__ == "__main__":
    input_embeddings, target_embeddings, neighbor_embeddings, edge_types, mask_neighbor = get_embeddings(path.join("..", "..", "data", "pems"))
    idx = random.randint(0, 10000)
    print(input_embeddings[idx, :, 0])
    print(target_embeddings[idx])
    print(neighbor_embeddings[idx, :, :, 0].t())
    # pickle.dump(neighbor_embeddings/10, open(path.join(BASE_DIR, "simple_neighbor_embeddings.bin"), "wb"))
