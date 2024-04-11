'''
PINNs for Inverse VIV Problem
@Author: Xuhui Meng
@Email: xuhui_meng@hust.edu.cn
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import tensorflow.compat.v1 as tf
import scipy.io as io
import numpy as np
import time
import matplotlib.pyplot as plt

from dataset import Dataset
from net import DNN

tf.disable_v2_behavior()

np.random.seed(1234)
tf.set_random_seed(1234)

#size of the DNN
layers = [1] + 4*[32] + [1]

def main():
    t_range = [0.0625, 10]
    NT = 100
    N_train = 150

    data = Dataset(t_range, NT, N_train)
    #inputdata
    t_u, u_data, t_f, f_data, t_ref, u_ref, f_ref = data.build_data()
    tmin, tmax = t_range[0], t_range[1]

    t_u_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    u_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    t_f_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    f_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    #physics-infromed neural networks
    pinn = DNN()
    W, b = pinn.hyper_initial(layers)
    k1 = tf.Variable(-2, dtype=tf.float32)
    k2 = tf.Variable(0., dtype=tf.float32)
    u_pred = pinn.fnn(t_u_train, W, b, tmin, tmax)
    f_pred = pinn.pdenn(t_f_train, W, b, k1, k2, tmin, tmax)

    loss = tf.reduce_mean(tf.square(f_pred - f_train)) + \
           tf.reduce_mean(tf.square(u_train - u_pred))

    train_adam = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    train_dict = {t_u_train: t_u, u_train: u_data, t_f_train: t_f, f_train: f_data}
    n = 0
    nmax = 30000
    start_time = time.perf_counter()
    while n <= nmax:
        n += 1
        u_, loss_, k1_, k2_, _ = sess.run([u_pred, loss, k1, k2, train_adam], feed_dict=train_dict)
        if n%100 == 0:
            print('Steps: %d, loss: %.3e, k1: %.3f, k2: %.3f'%(n, loss_, np.exp(k1_), np.exp(k2_)))
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))

    NT_test = 1000
    t_test = np.linspace(t_range[0], t_range[1], NT_test).reshape((-1, 1))
    u_test, f_test = sess.run([u_pred, f_pred], feed_dict={t_u_train: t_test, t_f_train: t_test})
    k1_, k2_ = sess.run([k1, k2])
    k1_ = np.exp(k1_)
    k2_ = np.exp(k2_)
    print('lambda1: %.3e, lambda2: %.3f'%(k1_, k2_))


    plt.figure()
    plt.plot(t_u, u_data, 'bo')
    plt.plot(t_ref, u_ref, 'k-')
    plt.plot(t_test, u_test, 'r--')
    plt.show()

    plt.figure()
    plt.plot(t_f, f_data, 'bo')
    plt.plot(t_ref, f_ref, 'k-')
    plt.plot(t_test, f_test, 'r--')
    plt.show()


    save_dict = {'t_u_train': t_u, 'u_train': u_data, 't_f_train': t_f, 'f_train': f_data, \
                 't_test': t_test, 'u_test': u_test, 'f_test': f_test, 'k1': k1_, 'k2': k2_}
    io.savemat('./Output/VIV_Pred.mat', save_dict)



if __name__ == '__main__':
    main()
