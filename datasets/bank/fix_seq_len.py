import torch
from os import path

BASE_DIR = path.join("..", "..", "data", "customers")

if __name__ == "__main__":
    seq_len = torch.load(path.join(BASE_DIR, "neighbors_msk.pt")).byte()
    seq_len = seq_len.sum(dim=1) + 1
    torch.save(seq_len, path.join(BASE_DIR, "ngh_msk.pt"))

