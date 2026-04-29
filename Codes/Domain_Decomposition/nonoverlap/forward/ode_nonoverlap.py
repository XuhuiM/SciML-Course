import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot
import time
import scipy.io as io
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)

from net import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on: %s'%(device))

def generate_data():
    data = io.loadmat('./Data/ode_data')
    t, u, f = data['t'].reshape((-1, 1)), data['u'].reshape((-1, 1)), data['f'].reshape((-1, 1))
    
    lb, le = 0, 50
    rb, re = le, t.shape[0] 

    t_l, u_l, t_r, u_r = t[lb:(le+1), :], u[lb:(le+1), :], t[rb:, :], u[rb:, :]
    t_l_bc, t_r_bc = t[le:(le+1), :], t[rb:(rb+1), :]

    num_u_train, num_f_train = 10, 25

    #training data for domain I
    '''
    idx_u_l = np.random.choice(le-lb, num_u_train, replace=False)
    t_u_l_train, u_l_train = t[idx_u_l], u[idx_u_l]
    '''
    t_u_l_train, u_l_train = t[lb], u[lb]
    idx_f_l = np.random.choice(le-lb, num_f_train, replace=False)
    t_f_l_train, f_l_train = t[idx_f_l], f[idx_f_l]

    #training data for domain II
    '''
    idx_u_r = np.random.choice(re - rb, num_u_train, replace=False) + rb
    t_u_r_train, u_r_train = t[idx_u_r], u[idx_u_r]
    '''
    t_u_r_train, u_r_train = t[re-1], u[re-1]

    idx_f_r = np.random.choice(re - rb, num_f_train, replace=False) + rb
    t_f_r_train, f_r_train = t[idx_f_r], f[idx_f_r]

    return t_u_l_train, u_l_train, t_f_l_train, f_l_train, \
           t_u_r_train, u_r_train, t_f_r_train, f_r_train, \
           t_l, u_l, t_r, u_r, t_l_bc, t_r_bc, \
           t, u, f
    '''
    num_u_train, num_f_train = 20, 50
    idx_u = np.random.choice(t.shape[0], num_u_train, replace=False)
    t_u_train, u_train = t[idx_u], u[idx_u]
    idx_f = np.random.choice(t.shape[0], num_f_train, replace=False)
    t_f_train, f_train = t[idx_f], f[idx_f]
    return t_u_train, u_train, t_f_train, f_train, t, u, f
    '''


if __name__ == '__main__':

    t_u_l_train, u_l_train, t_f_l_train, f_l_train, \
    t_u_r_train, u_r_train, t_f_r_train, f_r_train, \
    t_l_ref, u_l_ref,  t_r_ref, u_r_ref, t_l_bc, t_r_bc, \
    t_ref, u_ref, f_ref = generate_data()


    t_u_l_train, u_l_train = torch.tensor(t_u_l_train, dtype=torch.float32, requires_grad=False), torch.tensor(u_l_train, dtype=torch.float32)
    t_f_l_train, f_l_train = torch.tensor(t_f_l_train, dtype=torch.float32, requires_grad=True), torch.tensor(f_l_train, dtype=torch.float32)
    t_u_r_train, u_r_train = torch.tensor(t_u_r_train, dtype=torch.float32, requires_grad=False), torch.tensor(u_r_train, dtype=torch.float32)
    t_f_r_train, f_r_train = torch.tensor(t_f_r_train, dtype=torch.float32, requires_grad=True), torch.tensor(f_r_train, dtype=torch.float32)
    t_l_bc, t_r_bc = torch.tensor(t_l_bc, dtype=torch.float32, requires_grad=True), torch.tensor(t_r_bc, dtype=torch.float32, requires_grad=True)

    layers = [1] + [20]*2 + [1]

    model_l = FNN(layers)
    model_r = FNN(layers)

    #model = model.to(device)

    #a = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    a = torch.tensor(1.5, dtype=torch.float32)

    params = list(model_l.parameters()) + list(model_r.parameters())

    #loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(params, lr=1.0e-3)

    nmax = 10000
    n = 0
    while n < nmax:
        n += 1

        u_l_pred = model_l(t_u_l_train)
        u_l_bc_pred = model_l(t_l_bc)
        u_grad_l_bc_pred = torch.autograd.grad(model_l(t_l_bc), t_l_bc)[0]
        loss_data_l = torch.mean((u_l_pred - u_l_train)**2)
        R_l = eqnn(model_l, t_f_l_train, f_l_train, a)
        loss_eq_l = torch.mean(R_l**2)

        u_r_pred = model_r(t_u_r_train)
        u_r_bc_pred = model_r(t_r_bc)
        u_grad_r_bc_pred = torch.autograd.grad(model_r(t_r_bc), t_r_bc)[0]
        loss_data_r = torch.mean((u_r_pred - u_r_train)**2)
        R_r = eqnn(model_r, t_f_r_train, f_r_train, a)
        loss_eq_r = torch.mean(R_r**2)

        loss_coupling = torch.mean((u_l_bc_pred - u_r_bc_pred)**2) + torch.mean((u_grad_l_bc_pred - u_grad_r_bc_pred)**2)
        loss = loss_data_l + loss_eq_l + loss_data_r + loss_eq_r + loss_coupling

        opt.zero_grad()
        loss.backward()
        opt.step()

        if n%1000 == 0:
            print('Steps: %d, loss: %.3e, a: %.3e'%(n, loss.item(), a.item()))

    u_l_test = model_l(torch.tensor(t_l_ref, dtype=torch.float32))
    u_r_test = model_r(torch.tensor(t_r_ref, dtype=torch.float32))


    plt.figure()
    plt.plot(t_u_l_train.numpy(), u_l_train.numpy(), 'bo')
    plt.plot(t_u_r_train.numpy(), u_r_train.numpy(), 'bX')
    plt.plot(t_ref, u_ref, 'k-')
    plt.plot(t_l_ref, u_l_test.detach().numpy(), 'r--')
    plt.plot(t_r_ref, u_r_test.detach().numpy(), 'm--')
    plt.show()
