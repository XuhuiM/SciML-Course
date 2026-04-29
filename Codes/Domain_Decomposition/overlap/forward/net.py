import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    def __init__(self, layers, actn = nn.Tanh()):
        super().__init__()
        L = len(layers)
        self.linear = nn.ModuleList([nn.Linear(layers[l-1], layers[l]) for l in range(1, L)])
        self.activation = actn

    def forward(self, inputs):
        x = inputs
        for linear in self.linear[:-1]:
            x = self.activation(linear(x))
        x = self.linear[-1](x)
        return x

def eqnn(model, t_f_train, f_train, a):
    u_pred = model(t_f_train)
    u_t = torch.autograd.grad(u_pred, t_f_train,
               grad_outputs=torch.ones_like(t_f_train),
               create_graph=True,
            )[0]
    s = a*u_pred*(1. - u_pred)
    R = u_t - f_train - s
    return R
