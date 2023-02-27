'''
# Demo codes for sgd 
# Author: xuhui_meng@hust.edu.cn
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

np.random.seed(1234)
tf.set_random_seed(1234)

def main():
    '''
    num_train = 11
    #x = np.linspace(-1, 1, num_train).reshape((-1, 1))
    x = np.array([1.0]).reshape((-1, 1))
    y = 2*x

    Xmin = x.min(0)
    Xmax = x.max(0)
    '''

    x_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    W = tf.Variable(-3., dtype=tf.float32)
    loss = W**4 - 5*W**2 - 3*W


    '''
    train = tf.train.GradientDescentOptimizer(learning_rate=1.0e-3).minimize(loss)
    '''
    lr = 1.0e-2
    grad = tf.gradients(loss, W)[0]
    dW = -lr*(grad - 10.*tf.random.normal(shape=[1, 1])[0, 0])
    #dW = -lr*grad + 0.1*tf.random.normal(shape=[1, 1])[0, 0]
    train = W.assign_add(dW)
    

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    '''
    num_test = 101
    x_test = np.linspace(-1, 1, num_test).reshape((-1, 1))
    '''

    loss_list = []
    y_test_list = []
    w_val_list = []
    nmax = 10000
    n = 0
    while n <= nmax:
        loss_, _ = sess.run([loss, train])
        if n%200 == 0:
            '''
            loss_ = sess.run(loss, feed_dict={x_train: x, y_train: y})
            y_test_ , w_val = sess.run([y_pred, W], feed_dict={x_train: x_test})
            loss_list.append(loss_)
            y_test_list.append(y_test_)
            '''
            w_val, loss_ = sess.run([W, loss])
            w_val_list.append(w_val)
            loss_list.append(loss_)
            print('n: %d, w: %.3f, loss: %.3e'%(n, w_val, loss_))
        n += 1
    
    loss_dict = np.asarray(loss_list)
    '''
    y_test_dict = np.asarray(y_test_list)
    '''
    w_val_dict = np.array(w_val_list)
    #save_dict = {'x_train': x, 'y_train': y, 'x_test': x_test, 'y_test': y_test_dict, 'loss': loss_dict, 'w': w_val_dict}
    save_dict = {'w': w_val_dict, 'loss': loss_dict}
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
