import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class FNN(nn.Module):
    def __init__(self, layers, actn = nn.Tanh()):
        super().__init__()
        L = len(layers)
        self.linear = nn.ModuleList([nn.Linear(layers[l-1], layers[l]) for l in range(1, L)])
        self.activation = actn

    def forward(self, t, tmin, tmax):
        #t = self.time_encoder(t)
        X = 2.0*(t - tmin)/(tmax - tmin) - 1.0
        for linear in self.linear[:-1]:
            X = self.activation(linear(X))
        X = self.linear[-1](X)
        return X
