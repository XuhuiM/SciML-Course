import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot
import time
import scipy.io as io
import matplotlib.pyplot as plt

torch.manual_seed(1234)
np.random.seed(1234)

from net import FNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on: %s'%(device))

def generate_data():
    num_test, num_train = 101, 50
    x = np.linspace(-1, 1, num_test).reshape((-1, 1))
    y = np.sin(3*x)**3 
    idx = np.random.choice(num_test, num_train,  replace=False)
    x_train, y_train = x[idx], y[idx]
    return x_train, y_train, x, y


if __name__ == '__main__':

    x_train, y_train, x_ref, y_ref = generate_data()
    x_train, y_train = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    x_train, y_train = x_train.to(device), y_train.to(device)

    layers = [1] + [20]*2 + [1]

    model = FNN(layers)

    '''
    print(model.state_dict()['linear.2.weight'].shape)
    '''

    model = model.to(device)

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1.0e-3)

    nmax = 10000
    n = 0
    while n <= nmax:
        n += 1
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if n%1000 == 0:
            print('Steps: %d, loss: %.3e'%(n, loss.item()))

    y_test = model(torch.tensor(x_ref, dtype=torch.float32))

    plt.figure()
    plt.plot(x_train.numpy(), y_train.numpy(), 'bo')
    plt.plot(x_ref, y_ref, 'k-')
    plt.plot(x_ref, y_test.detach().numpy(), 'r--')
    plt.show()
