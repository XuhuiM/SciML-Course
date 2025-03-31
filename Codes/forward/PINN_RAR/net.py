import tensorflow as tf
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

    def fnn(self, X, W, b):
        A = X
        L = len(W)
        for i in range(L-1):
            A = tf.tanh(tf.add(tf.matmul(A, W[i]), b[i]))
        Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        
        return Y

    def pdenn(self, X, W_u, b_u):
        u = self.fnn(X, W_u, b_u)
        u_x = tf.gradients(u, X)[0]
        u_xx = tf.gradients(u_x, X)[0]
        f = tf.sin(X) + 2*tf.sin(2*X) + 3*tf.sin(3*X) + 4*tf.sin(4*X) + 8*tf.sin(8*X)
        r = f + u_xx
        f_pred = -u_xx

        return f_pred, r
