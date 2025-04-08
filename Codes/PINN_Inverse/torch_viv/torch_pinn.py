import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import time
import scipy.io as io
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)

from net import FNN
from dataset import TrainData

layers = [1] + [32]*2 + [1]

def eqnn(model, t, tmin, tmax, k1, k2):
    u = model(t, tmin, tmax)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(t), create_graph=True)[0]
    rho = 2.0
    k1_, k2_ = torch.exp(k1), torch.exp(k2)
    f = rho*u_tt + k1_*u_t + k2_*u
    return f

def main():

    #training data
    t_range = [0.0625, 10]
    NT = 100
    N_train = 150
    data = TrainData(t_range, NT, N_train)

    #training data
    t_u, u_data, t_f, f_data, t_ref, u_ref, f_ref = data.build_data()
    tmin, tmax = t_range[0],  t_range[1]
    t_u, u_data = torch.tensor(t_u, dtype=torch.float32), torch.tensor(u_data, dtype=torch.float32)
    t_f, f_data = torch.tensor(t_f, dtype=torch.float32, requires_grad=True), torch.tensor(f_data, dtype=torch.float32)
    k1, k2 = torch.tensor([-2], dtype=torch.float32, requires_grad=True), torch.tensor([0.], dtype=torch.float32, requires_grad=True)
    k1, k2 = torch.nn.Parameter(k1), torch.nn.Parameter(k2)

    #create DNNs
    model = FNN(layers)

    model.register_parameter('k1', k1)
    model.register_parameter('k2', k2)
    #define loss and optimizer
    opt = optim.Adam(model.parameters(), lr=1.0e-3)

    nmax = 30000
    n = 0
    while n < nmax:
        n += 1

        u_pred = model(t_u, tmin, tmax)
        f_pred = eqnn(model, t_f, tmin, tmax, k1, k2)
        loss = torch.mean(torch.square(u_pred - u_data)) + torch.mean(torch.square(f_pred - f_data))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if n%1000 == 0:
            print('Steps: %d, loss: %.3e, k1: %.3f, k2: %.3f'%(n, loss.item(), torch.exp(k1).item(), torch.exp(k2).item()))

    NT_test = 1000
    t_test = torch.linspace(t_range[0], t_range[1], NT_test).reshape((-1, 1))
    t_f_test = t_test
    t_f_test = t_f_test.requires_grad_(True)
    u_test, f_test = model(t_test, tmin, tmax), eqnn(model, t_f_test, tmin, tmax, k1, k2)
    k1_, k2_ = torch.exp(k1).item(), torch.exp(k2).item()
    print('lambda1: %.3e, lambda2: %.3e'%(k1_, k2_))


    plt.figure()
    plt.plot(t_u.detach().numpy(), u_data.detach().numpy(), 'bo')
    plt.plot(t_ref, u_ref, 'k-')
    plt.plot(t_test.detach().numpy(), u_test.detach().numpy(), 'r--')
    plt.show()

    plt.figure()
    plt.plot(t_f.detach().numpy(), f_data.detach().numpy(), 'bo')
    plt.plot(t_ref, f_ref, 'k-')
    plt.plot(t_test.detach().numpy(), f_test.detach().numpy(), 'r--')
    plt.show()


    '''
    save_dict = {'t_u_train': t_u, 'u_train': u_data, 't_f_train': t_f, 'f_train': f_data, \
                 't_test': t_test, 'u_test': u_test, 'f_test': f_test, 'k1': k1_, 'k2': k2_}
    io.savemat('./Output/VIV_Pred.mat', save_dict)
    '''


if __name__ == '__main__':

    main()
