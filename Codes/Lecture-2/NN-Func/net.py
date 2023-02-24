import tensorflow as tf
import numpy as np

class DNN:
    def __init__(self, layer_size):
        self.size = layer_size
    
    def hyper_initial(self):
        L = len(self.size)
        W = []
        b = []
        for l in range(1, L):
            in_dim = self.size[l-1]
            out_dim = self.size[l]
            std = np.sqrt(2/(in_dim + out_dim))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            W.append(weight)
            b.append(bias)

        return W, b

    def fnn(self, X, W, b, Xmin, Xmax, actn=tf.tanh):
        A = 2.0*(X - Xmin)/(Xmax - Xmin) - 1.0
        L = len(W)
        for i in range(L-1):
            A = actn(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y
