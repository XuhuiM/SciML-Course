import tensorflow.compat.v1 as tf
import numpy as np

class DNN:
    def __init__(self, layers, w_0, w_1, w_2, b_0, b_1, b_2): #, Xmin, Xmax):
        '''
        self.Xmin = Xmin
        self.Xmax = Xmax
        '''
        self.layers = layers
        self.w_0 = w_0
        self.w_1 = w_1
        self.w_2 = w_2
        self.b_0 = b_0
        self.b_1 = b_1
        self.b_2 = b_2

    '''
    def hyper_initial(self):
        L = len(self.layers)
        Weights = []
        Biases = []
        for l in range(1, L):
            in_dim = self.layers[l-1]
            out_dim = self.layers[l]
            std = np.sqrt(2/(in_dim+out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            Weights.append(weight)
            Biases.append(bias)

        return Weights, Biases
    '''

    def fnn(self, X):
        A = X
        A = tf.tanh(tf.add(tf.matmul(A, self.w_0), self.b_0))
        A = tf.tanh(tf.add(tf.matmul(A, self.w_1), self.b_1))
        Y = tf.add(tf.matmul(A, self.w_2), self.b_2)
        return Y

    def fnn_output(self, X):
        A = X
        A = tf.tanh(tf.add(tf.einsum('ij,bjk->bik', A, self.w_0), self.b_0))
        A = tf.tanh(tf.add(tf.einsum('bij,bjk->bik', A, self.w_1), self.b_1))
        Y = tf.add(tf.einsum('bij,bjk->bik', A, self.w_2), self.b_2)
        return Y
