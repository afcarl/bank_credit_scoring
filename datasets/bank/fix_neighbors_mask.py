import torch
from os import path

BASE_DIR = path.join("..", "..", "data", "customers")

if __name__ == "__main__":
    neigh_len = torch.load(path.join(BASE_DIR, "neighbors_msk.pt")).byte()
    mask_neighbor = torch.ByteTensor(neigh_len.size(0), 10).fill_(1)

    for idx, value in enumerate(neigh_len.squeeze().numpy()):
        value -= 1
        if value > 0:
            mask_neighbor[idx, :value] = 0

    torch.save(mask_neighbor, path.join(BASE_DIR, "mask_neighbor.pt"))

