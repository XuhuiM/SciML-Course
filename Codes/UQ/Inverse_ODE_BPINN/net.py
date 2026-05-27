import tensorflow.compat.v1 as tf
import numpy as np

class DNN:
    def __init__(self): #, Xmin, Xmax):
        '''
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.layers = layers
        self.w_0 = w_0
        self.w_1 = w_1
        self.w_2 = w_2
        self.b_0 = b_0
        self.b_1 = b_1
        self.b_2 = b_2
        '''
        pass

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

    def fnn(self, X, W, b):
        inp = tf.convert_to_tensor(X, dtype=tf.float32)
        Y = 2.*inp - 1.0
        L = len(W)
        for i in range(L-1):
            Y = tf.tanh(tf.add(tf.matmul(Y, W[i]), b[i]))
        Y = tf.add(tf.matmul(Y, W[-1]), b[-1])
        return inp, Y

    def pde(self, X, W_u, b_u, a):
        t, u = self.fnn(X, W_u, b_u)
        u_t = tf.gradients(u, t)[0]
        #_, s = self.fnn(X, W_s, b_s)
        s = a*u*(1. - u)
        f = u_t - s
        return f

    def fnnout(self, X, W, b):
        bs = tf.shape(W[0])[0]
        inp = tf.tile(X[None, :, :], [bs, 1, 1])
        Y = 2.*inp - 1.
        L = len(W)
        for i in range(L-1):
            Y = tf.tanh(tf.add(tf.einsum('bij,bjk->bik', Y, W[i]), b[i]))
        Y = tf.add(tf.einsum('bij,bjk->bik', Y, W[-1]), b[-1])
        return inp, Y

    def pdeout(self, X, W_u, b_u, a):
        '''
        bs = tf.shape(W_u[0])[0]
        T = tf.tile(X[None, :, :], [bs, 1, 1])
        '''
        T, u = self.fnnout(X, W_u, b_u)
        u_t = tf.gradients(u, T)[0]
        #_, s = self.fnnout(X, W_s, b_s)
        s = a*u*(1. - u)
        f = u_t - s
        return f
