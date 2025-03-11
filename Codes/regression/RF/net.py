import tensorflow as tf
import numpy as np

class DNN:
    def __init__(self):
        pass
    
    def hyper_initial(self, layers_size, R, x_col):
        L = len(layers_size)
        x_col = tf.convert_to_tensor(x_col, dtype=tf.float32)
        W = []
        b = []
        for l in range(1, L):
            in_dim = layers_size[l-1]
            out_dim = layers_size[l]
            std = np.sqrt(2/(in_dim + out_dim))
            weight_init = tf.random.uniform(shape=[in_dim, out_dim], minval=-R, maxval=R, dtype=tf.float32)
            '''
            bias_init = tf.random.uniform(shape=[1, out_dim], minval=-R, maxval=R, dtype=tf.float32)
            '''
            if l == 1:
                bias_init = -tf.multiply(weight_init, x_col)
            else:
                bias_init = tf.random.uniform(shape=[1, out_dim], minval=-R, maxval=R, dtype=tf.float32)
            weight = tf.Variable(weight_init)
            bias = tf.Variable(bias_init)
            '''
            bias = tf.Variable(tf.zeros(shape=[1, out_dim], dtype=tf.float32))
            weight = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=std, dtype=tf.float32))
            bias = tf.Variable(tf.zeros(shape=[1, out_dim], dtype=tf.float32))
            '''
            W.append(weight)
            b.append(bias)

        return W, b

    def fnn(self, X, W, b, Xmin, Xmax, actn=tf.tanh, is_linear=False):
        #A = 2.0*(X - Xmin)/(Xmax - Xmin) - 1.0
        feature = []
        L = len(W)
        A = X
        for i in range(L):
            A = actn(tf.add(tf.matmul(A, W[i]), b[i]))
            feature.append(A)
        if is_linear:
            Y = tf.add(tf.matmul(A, W[-1]), b[-1])
        else:
            Y = A
        return Y, feature
