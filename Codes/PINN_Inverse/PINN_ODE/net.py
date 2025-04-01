import tensorflow.compat.v1 as tf
import numpy as np

class DNN:
    def __init__(self):
        pass
    
    def hyper_initial(self, layers):
        L = len(layers)
        Weights = []
        Biases = []
        for l in range(1, L):
            in_dim = layers[l-1]
            out_dim = layers[l]
            std = np.sqrt(2/(in_dim + out_dim))
            weight = tf.Variable(tf.random.normal(shape=[in_dim, out_dim], stddev=std))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim]))
            Weights.append(weight)
            Biases.append(bias)

        return Weights, Biases

    def fnn(self, X, W, b, Xmin, Xmax):
        A = 2*(X - Xmin)/(Xmax - Xmin) - 1
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y

    def pdenn(self, t, W, b, k, Xmin, Xmax):
        u = self.fnn(t, W, b, Xmin, Xmax)
        u_t = tf.gradients(u, t)[0]
        f = u_t/k

        return f
