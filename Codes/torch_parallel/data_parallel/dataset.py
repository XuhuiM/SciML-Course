import torch
from torch.utils.data import Dataset

class TrainData(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, idx):
        inputs = self.x_train[idx]
        targets = self.y_train[idx]
        return inputs, targets

    def __len__(self):
        return self.x_train.shape[0]

