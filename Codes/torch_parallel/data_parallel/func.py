import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.utils.data as Data

import numpy as np
import time
import scipy.io as io
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)

from net import FNN
from dataset import TrainData

layers = [1] + [20]*2 + [1]

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

def generate_data():
    num_test, num_train = 101, 64
    x = np.linspace(-1, 1, num_test).reshape((-1, 1))
    y = np.sin(3*x)**3 
    idx = np.random.choice(num_test, num_train,  replace=False)
    x_train, y_train = x[idx], y[idx]
    return x_train, y_train, x, y


def main(rank, world_size, train_dataset, test_dataset):

    #set up distributed devices
    ddp_setup(rank, world_size)
    print('rank id: %d'%(rank))

    #training data
    x_train, y_train = train_dataset[0], train_dataset[1]
    bs = x_train.shape[0]//world_size
    train_inputs = x_train[rank*bs:(rank+1)*bs, :]
    train_targets = y_train[rank*bs:(rank+1)*bs, :]

    #create DNNs
    model = FNN(layers)
    model = DDP(model) 

    #define loss and optimizer
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=1.0e-3)

    nmax = 10000
    n = 0
    while n < nmax:
        n += 1
        y_pred = model(train_inputs)
        loss = loss_fn(y_pred, train_targets)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if n%100 == 0:
            print('Steps: %d, loss: %.3e'%(n, loss.item()))

    dist.destroy_process_group()

    x_ref, y_ref = test_dataset[0], test_dataset[1]
    y_test = model(torch.tensor(x_ref, dtype=torch.float32))
    save_dict = {'x_train': train_inputs.numpy(), 'y_train': train_targets.numpy(), 'x_test': x_ref, 'y_pred': y_test.detach().numpy(), 'y_ref': y_ref}
    filename = './Output/y_pred_' + str(rank) + '.mat'
    io.savemat(filename, save_dict)


if __name__ == '__main__':

    x_train, y_train, x_ref, y_ref = generate_data()
    x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    train_dataset = [x_train, y_train]
    test_dataset = [x_ref, y_ref]

    #set up distributed devices
    world_size = 2
    mp.spawn(main, args=(world_size, train_dataset, test_dataset), nprocs=world_size)
