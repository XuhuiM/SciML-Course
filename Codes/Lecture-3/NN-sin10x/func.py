'''
# Demo codes for NNs with underfitting/overfitting
# Author: xuhui_meng@hust.edu.cn
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

from net import DNN 

layers = [1] + 4*[20] + [1]

np.random.seed(1234)
tf.set_random_seed(1234)

def main():
    num_train = 21
    x = np.linspace(-1, 1, num_train).reshape((-1, 1))
    y = np.sin(16*x)

    Xmin = x.min(0)
    Xmax = x.max(0)

    x_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    model = DNN(layers)
    W, b = model.hyper_initial()
    y_pred = model.fnn(x_train, W, b, Xmin, Xmax)

    l2_loss = tf.add_n([tf.nn.l2_loss(w_) for w_ in W])
    loss = tf.reduce_mean(tf.square(y_pred - y_train)) + 1.0e-5*l2_loss
    train = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_test = 101
    x_test = np.linspace(-1, 1, num_test).reshape((-1, 1))

    loss_list = []
    y_test_list = []
    nmax = 50000
    n = 0
    while n <= nmax:
        y_pred_, loss_, _ = sess.run([y_pred, loss, train], feed_dict={x_train: x, y_train: y})
        if n%1000 == 0:
            loss_ = sess.run(loss, feed_dict={x_train: x, y_train: y})
            y_test_ = sess.run(y_pred, feed_dict={x_train: x_test})
            loss_list.append(loss_)
            y_test_list.append(y_test_)
            print('n: %d, loss: %.3e'%(n, loss_))
        n += 1
    
    loss_dict = np.asarray(loss_list)
    y_test_dict = np.asarray(y_test_list)
    save_dict = {'x_train': x, 'y_train': y, 'x_test': x_test, 'y_test': y_test_dict, 'loss': loss_dict}
    io.savemat('./Output/pred.mat', save_dict)

    '''
    x_test = np.linspace(-1, 1, 1001).reshape((-1, 1))
    y_test = sess.run(y_pred, feed_dict={x_train: x_test})
    plt.figure()
    plt.plot(x, y, 'k.')
    plt.plot(x_test, y_test, 'r-')
    plt.show()
    '''

if __name__ == '__main__':
    main()
