''''''''''''''''''''
@ Demo code for gradient descent
@Author: xuhui_meng@hust.edu.cn
''''''''''''''''''''
import os

os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

np.random.seed(1234)
tf.set_random_seed(1234)

def main():
    num_train = 11
    #x = np.linspace(-1, 1, num_train).reshape((-1, 1))
    x = np.array([1.0]).reshape((-1, 1))
    y = 2*x

    Xmin = x.min(0)
    Xmax = x.max(0)

    x_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    W = tf.Variable(5.0, dtype=tf.float32)
    y_pred = W*x_train

    loss = tf.reduce_mean(tf.square(y_pred - y_train))
    train = tf.train.GradientDescentOptimizer(learning_rate=0.98).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_test = 101
    x_test = np.linspace(-1, 1, num_test).reshape((-1, 1))

    loss_list = []
    y_test_list = []
    w_val_list = []
    nmax = 100
    n = 0
    while n <= nmax:
        y_pred_, loss_, _ = sess.run([y_pred, loss, train], feed_dict={x_train: x, y_train: y})
        if n%1 == 0:
            loss_ = sess.run(loss, feed_dict={x_train: x, y_train: y})
            y_test_ , w_val = sess.run([y_pred, W], feed_dict={x_train: x_test})
            loss_list.append(loss_)
            y_test_list.append(y_test_)
            w_val_list.append(w_val)
            print('n: %d, w: %.3f, loss: %.3e'%(n, w_val, loss_))
        n += 1
    
    loss_dict = np.asarray(loss_list)
    y_test_dict = np.asarray(y_test_list)
    w_val_dict = np.array(w_val_list)
    save_dict = {'x_train': x, 'y_train': y, 'x_test': x_test, 'y_test': y_test_dict, 'loss': loss_dict, 'w': w_val_dict}
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
