import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as io

np.random.seed(1234)

class DataSet:
    def __init__(self, N, batch_size):
        self.N = N
        self.batch_size = batch_size
        self.x_train, self.F_train, self.U_train, self.F_test, self.U_test, \
        self.u_train_mean, self.u_train_std = self.samples()

    def decode(self, x):
        return x*(self.u_train_std + 1.0e-6) + self.u_train_mean

    def samples(self):
        num_train = 2000
        num_test = 1000

        data = io.loadmat('./Data/ODE_Train_Data')
        F = data['F']
        U = data['U']
        x_train = data['x_train']

        F_train = F[:num_train, :]
        U_train = U[:num_train, :]
        F_test = F[-num_test:, :]
        U_test = U[-num_test:, :]

        f_train_mean = np.mean(F_train, axis=0, keepdims=True)
        f_train_std = np.std(F_train, axis=0, keepdims=True)
        u_train_mean = np.mean(U_train, axis=0, keepdims=True)
        u_train_std = np.std(U_train, axis=0, keepdims=True)

        F_train = (F_train - f_train_mean)/(f_train_std + 1.0e-6)
        U_train = (U_train - u_train_mean)/(u_train_std + 1.0e-6)

        F_test = (F_test - f_train_mean)/(f_train_std + 1.0e-6)

        return x_train, F_train, U_train, F_test, U_test, u_train_mean, u_train_std
               
    def minibatch(self):
        batch_id = np.random.choice(self.F_train.shape[0], self.batch_size, replace=False)
        f_train = [self.F_train[i:i+1] for i in batch_id]
        f_train = np.concatenate(f_train, axis=0)
        u_train = [self.U_train[i:i+1] for i in batch_id]
        u_train = np.concatenate(u_train, axis=0)

        return self.x_train, f_train, u_train

    def testbatch(self, num_test):
        batch_id = np.random.choice(self.F_test.shape[0], num_test, replace=False)
        f_test = [self.F_test[i:i+1] for i in batch_id]
        f_test = np.concatenate(f_test, axis=0)
        u_test = [self.U_test[i:i+1] for i in batch_id]
        u_test = np.concatenate(u_test, axis=0)
        batch_id = np.reshape(batch_id, (-1, 1))

        return batch_id, self.x_train, f_test, u_test
