import os
#os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot
import time
import scipy.io as io
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)

from net import *

'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on: %s'%(device))
'''

def env_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.cuda.set_device(rank%torch.cuda.device_count())
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def generate_data():
    data = io.loadmat('../Data/ode_data')
    t, u, f = data['t'].reshape((-1, 1)), data['u'].reshape((-1, 1)), data['f'].reshape((-1, 1))
    
    lb, le = 0, 60
    rb, re = 40, t.shape[0] 

    num_bc = 2
    t_l, u_l, t_r, u_r = t[lb:(le+1), :], u[lb:(le+1), :], t[rb:, :], u[rb:, :]
    t_l_bc, t_r_bc = t[(le-num_bc):(le+1), :], t[rb:(rb+num_bc+1), :]

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

def main(world_size, train_dataset_0, train_dataset_1, test_dataset_0, test_dataset_1, test_dataset):

    #set up distributed devices
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank%torch.cuda.device_count())
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    '''
    env_setup(rank, world_size)
    print('rank id: %d'%(rank))
    '''

    device = torch.device('cuda', rank)
    print(device)

    if rank == 0:
        t_u_train, u_train = train_dataset_0[0].to(device), train_dataset_0[1].to(device)
        t_f_train, f_train = train_dataset_0[2].to(device), train_dataset_0[3].to(device)
        t_l_bc, t_r_bc = train_dataset_0[4].to(device), train_dataset_0[-1].to(device)
        #t_coupling_bc = train_dataset_0[-1]
        t_sub_ref, u_sub_ref = test_dataset_0[0], test_dataset_0[1]
        t_ref, u_ref = test_dataset[0], test_dataset[1]
    else:
        t_u_train, u_train = train_dataset_1[0].to(device), train_dataset_1[1].to(device)
        t_f_train, f_train = train_dataset_1[2].to(device), train_dataset_1[3].to(device)
        t_l_bc, t_r_bc = train_dataset_1[-1].to(device), train_dataset_1[4].to(device)
        #t_coupling_bc = train_dataset_1[4]
        t_sub_ref, u_sub_ref = test_dataset_1[0], test_dataset_1[1]
        t_ref, u_ref = test_dataset[0], test_dataset[1]

    layers = [1] + [20]*2 + [1]

    model = FNN(layers)
    model = model.to(device)

    #a = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    a = torch.tensor(1.5, dtype=torch.float32, requires_grad=False).to(device)

    params = list(model.parameters())

    #loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(params, lr=1.0e-3)

    '''
    b = 2*a
    print(b)
    if rank==0:
        dist.isend(a, dst=(rank+1)%world_size).wait()
        dist.irecv(b, src=(rank+1)%world_size).wait()
    else:
        dist.irecv(b, src=(rank+1)%world_size).wait()
        dist.isend(a, dst=(rank+1)%world_size).wait()
    print(b)
    return
    '''

    nmax = 20000
    n = 0
    while n < nmax:
        n += 1

        u_pred = model(t_u_train)
        loss_data = torch.mean((u_pred - u_train)**2)
        R = eqnn(model, t_f_train, f_train, a)
        loss_eq = torch.mean(R**2)

        u_coupling_bc_pred = model(t_l_bc)
        u_bc_to_be_sent = model(t_r_bc)
        u_coupling_bc_ref = torch.zeros_like(u_coupling_bc_pred)

        '''
        u_grad_coupling_bc_ref = torch.zeros_like(u_grad_coupling_bc_pred)
        '''

        if rank == 0:
            dist.isend(u_bc_to_be_sent, dst=(rank+1)%world_size).wait()
            #dist.isend(u_grad_coupling_bc_pred, dst=(rank+1)%world_size).wait()
            dist.irecv(u_coupling_bc_ref, src=(rank+1)%world_size).wait()
            #dist.irecv(u_grad_coupling_bc_ref, src=(rank+1)%world_size).wait()
        else:
            dist.irecv(u_coupling_bc_ref, src=(rank+1)%world_size).wait()
            #dist.irecv(u_grad_coupling_bc_ref, src=(rank+1)%world_size).wait()
            dist.isend(u_bc_to_be_sent, dst=(rank+1)%world_size).wait()
            #dist.isend(u_grad_coupling_bc_pred, dst=(rank+1)%world_size).wait()

        loss_coupling = torch.mean((u_coupling_bc_pred - u_coupling_bc_ref)**2)

        loss = loss_data + loss_eq + loss_coupling

        opt.zero_grad()
        loss.backward()
        opt.step()

        if n%100 == 0:
            print('Steps: %d, loss: %.3e, a: %.3e'%(n, loss.item(), a.item()))

    dist.destroy_process_group()

    u_test = model(torch.tensor(t_sub_ref, dtype=torch.float32).to(device))
    save_dict = {'t_sub_ref': t_sub_ref, 'u_test': u_test.cpu().detach().numpy(), \
                 't_u_train': t_u_train.cpu().detach().numpy(), 'u_train': u_train.cpu().detach().numpy(),\
                 't_ref': t_ref, 'u_ref': u_ref}
    filename = './Output/pred_' + str(rank) + '.mat'
    io.savemat(filename, save_dict)

    '''
    plt.figure()
    plt.plot(t_u_l_train.numpy(), u_l_train.numpy(), 'bo')
    plt.plot(t_u_r_train.numpy(), u_r_train.numpy(), 'bX')
    plt.plot(t_ref, u_ref, 'k-')
    plt.plot(t_l_ref, u_l_test.detach().numpy(), 'r--')
    plt.plot(t_r_ref, u_r_test.detach().numpy(), 'm--')
    plt.show()
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
    t_l_bc, t_r_bc = torch.tensor(t_l_bc, dtype=torch.float32, requires_grad=False), torch.tensor(t_r_bc, dtype=torch.float32, requires_grad=False)

    train_dataset_0 = [t_u_l_train, u_l_train, t_f_l_train, f_l_train, t_l_bc, t_r_bc]
    train_dataset_1 = [t_u_r_train, u_r_train, t_f_r_train, f_r_train, t_l_bc, t_r_bc]
    test_dataset_0 = [t_l_ref, u_l_ref]
    test_dataset_1 = [t_r_ref, u_r_ref]
    test_dataset = [t_ref, u_ref, f_ref]

    #set up distributed devices
    world_size = 2
    main(world_size, train_dataset_0, train_dataset_1, test_dataset_0, test_dataset_1, test_dataset)
    '''
    world_size = 2
    mp.spawn(main, args=(world_size, train_dataset_0, train_dataset_1, test_dataset_0, test_dataset_1, test_dataset), \
             nprocs=world_size, join=True)
    '''
