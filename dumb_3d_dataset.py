import torch
from torch.utils.data.dataset import Dataset


class Dumb3DDataset(Dataset):
    def __init__(self, length: int, num_classes: int):
        super().__init__()
        self.length = length
        self.num_classes = num_classes

    def __getitem__(self, _):
        return torch.zeros(1, 64, 64, 64), torch.zeros(self.num_classes, 64, 64, 64)

    def __len__(self):
        return self.length
