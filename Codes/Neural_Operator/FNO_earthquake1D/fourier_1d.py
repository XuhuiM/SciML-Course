"""
@author: Xuhui Meng
This file is adapted from the Fourier Neural Operator for 1D problem such as 
the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import os
import time
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
import scipy

# torch.manual_seed(0)
# np.random.seed(0)

print("\n=============================")
print("torch.cuda.is_available(): " + str(torch.cuda.is_available()) )
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
print("=============================\n")

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



def FNO_main(train_data_res, save_index=0):
    """
    Parameters
    ----------
    train_data_res : resolution of the training data
    save_index : index of the saving folder
    """
    
    ################################################################
    #  configurations
    ################################################################
    ntrain = 90
    ntest = 10
    
    s = train_data_res
    # sub = 2**6 #subsampling rate
    sub = 2500 // s  # subsampling rate (step size)
    
    batch_size = 10
    learning_rate = 0.001
    
    epochs = 100       # default 500
    step_size = 100     # default 100
    gamma = 0.5
    
    modes = 300
    width = 32
    
    ################################################################
    # read training data
    ################################################################
    
    # Data is of the shape (number of samples, grid size)
    dataloader = MatReader('Data/eq_data_N100_r2500.mat')
    x_data = dataloader.read_field('Load')[:,::sub]
    y_data = dataloader.read_field('Response')[:,::sub]
    
    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]
    
    # cat the locations information
    grid_all = dataloader.read_field('Time') / 50.   # normalized to [0, 1]
    grid = grid_all[::sub,:]
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,1), grid.repeat(ntrain,1,1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
    
    # model
    #model = FNO1d(modes, width).cuda()
    model = FNO1d(modes, width)
    print(count_params(model))
    
    
    ################################################################
    # training and evaluation
    ################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    start_time = time.time()
    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            #x, y = x.cuda(), y.cuda()
    
            optimizer.zero_grad()
            out = model(x)
    
            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            mse.backward()
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            # l2.backward() # use the l2 relative loss
    
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
    
        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                #x, y = x.cuda(), y.cuda()
                out = model(x)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()
    
        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest
    
        t2 = default_timer()
        print("Epoch: %d, time: %.3f, Train Loss: %.3e, Train l2: %.4f, Test l2: %.4f" 
                  % ( ep, t2-t1, train_mse, train_l2, test_l2) )
        # print(ep, t2-t1, train_mse, train_l2, test_l2)
    
    elapsed = time.time() - start_time
    print("\n=============================")
    print("Training done...")
    print('Training time: %.3f'%(elapsed))
    print("=============================\n")
    
    
    # ====================================
    # saving settings
    # ====================================
    # current_directory = os.getcwd()
    # resolution = "TrainRes_"+str(train_data_res)
    # folder_index = str(save_index)
    
    # results_dir = "/results/" + resolution +"/" + folder_index +"/"
    # save_results_to = current_directory + results_dir
    # if not os.path.exists(save_results_to):
    #     os.makedirs(save_results_to)
        
    # model_dir = "/model/" + resolution +"/" + folder_index +"/"
    # save_models_to = current_directory + model_dir
    # if not os.path.exists(save_models_to):
    #     os.makedirs(save_models_to)
    
    
    ################################################################
    # testing
    ################################################################
    # torch.save(model, save_models_to+'fourier_eq')
    # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    
    # pred = torch.zeros(y_test.shape)
    # index = 0
    # test_l2 = 0
    # t1 = default_timer()
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         x, y = x.cuda(), y.cuda()
    
    #         out = model(x)
    #         pred[index] = out.squeeze()
    
    #         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
    #         # print(index, test_l2)
    #         index = index + 1
    # t2 = default_timer()
    # testing_time = t2-t1
    
    # test_l2 = test_l2/index
    # print("\n=============================")
    # print('Testing error: %.3e'%(test_l2))
    # print("=============================\n")

    # scipy.io.savemat(save_results_to+'earthquake_test.mat', 
    #                   mdict={'x_test': dataloader.read_field('Load')[-ntest:,::sub].numpy(),
    #                         'y_test': y_test.numpy(), 
    #                         'y_pred': pred.cpu().numpy(),
    #                         'testing_time': testing_time})


if __name__ == "__main__":
    
    training_data_resolution = 2500
    FNO_main(training_data_resolution, save_index=0)
    
    
# mode = 300, width = 128
# l2 errors
# 0.0358, 0.0363, 0.0367, 0.0363, 0.0354
#   3.61±0.05\%



