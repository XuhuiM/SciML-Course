'''
PINNs for Inverse ODE Problem
@Author: Xuhui Meng
@Email: xuhui_meng@hust.edu.cn
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
import scipy.io as io
import numpy as np
import time
import matplotlib.pyplot as plt

from net import DNN

tf.disable_v2_behavior()

np.random.seed(1234)
tf.set_random_seed(1234)

#size of the DNN
layers = [1] + 4*[32] + [1]

def main():
    t_range = [0, 10]

    num_all = 1001
    t_all = np.linspace(t_range[0], t_range[1], num_all).reshape((-1, 1))
    u_all = t_all + np.sin(0.5*np.pi*t_all)
    f_all = 1 + 0.5*np.pi*np.cos(0.5*np.pi*t_all)

    tmin, tmax = t_range[0], t_range[1]

    NT = 10
    N_train = 500

    #training data for u and f
    u_id = np.random.choice(num_all//2, NT, replace=False)
    t_u, u_data = t_all[u_id], u_all[u_id]
    f_id = np.random.choice(num_all, N_train, replace=False)
    t_f, f_data = t_all[f_id], f_all[f_id]

    t_u_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    u_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    t_f_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    f_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    #physics-infromed neural networks
    pinn = DNN()
    W, b = pinn.hyper_initial(layers)
    k = tf.Variable(0.1, dtype=tf.float32)
    u_pred = pinn.fnn(t_u_train, W, b, tmin, tmax)
    f_pred = pinn.pdenn(t_f_train, W, b, k, tmin, tmax)

    loss = tf.reduce_mean(tf.square(f_pred - f_train)) + \
           tf.reduce_mean(tf.square(u_train - u_pred))

    train_adam = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    train_dict = {t_u_train: t_u, u_train: u_data, t_f_train: t_f, f_train: f_data}
    n = 0
    nmax = 10000
    start_time = time.perf_counter()
    while n <= nmax:
        n += 1
        u_, loss_, k_, _ = sess.run([u_pred, loss, k, train_adam], feed_dict=train_dict)
        if n%100 == 0:
            print('Steps: %d, loss: %.3e, k: %.3e'%(n, loss_, k_))
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))

    '''
    NT_test = 1000
    t_test = np.linspace(t_range[0], t_range[1], NT_test).reshape((-1, 1))
    '''
    t_test = t_all
    u_test, f_test, k_ = sess.run([u_pred, f_pred, k], feed_dict={t_u_train: t_test, t_f_train: t_test})
    
    t_ref, u_ref, f_ref = t_all, u_all, f_all

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
                 't_test': t_test, 'u_test': u_test, 'f_test': f_test, 'k': k_}
    io.savemat('./Output/ODE_Inverse_Pred.mat', save_dict)



if __name__ == '__main__':
    main()
