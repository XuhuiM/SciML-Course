import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import time
import scipy.io as io
import matplotlib.pyplot as plt

from net import DNN

np.random.seed(1234)
tf.set_random_seed(1234)

#size of the DNN
layers = [1] + 2*[50] + [1]

def training_data():
    num_f_train = 100
    x_f_train = np.linspace(-1., 1., num_f_train).reshape((-1, 1))
    f_train = np.pi**2*np.sin(np.pi*x_f_train)
    x_bc_l = np.array([-1.]).reshape((-1, 1))
    u_l = np.array([0.]).reshape((-1, 1))
    x_bc_r = np.array([ 1.]).reshape((-1, 1))
    u_r = np.array([0.]).reshape((-1, 1))
    x_u_train = np.vstack((x_bc_l, x_bc_r))
    u_train = np.vstack((u_l, u_r))

    return x_u_train, u_train, x_f_train, f_train

def main():

    #generating training data
    x_u_train, u_train, x_f_train, f_train = training_data()

    x_u_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    u_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    x_f_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    f_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    
    #physics-infromed neural networks
    model = DNN()
    W_u, b_u = model.hyper_initial(layers)
    u_pred = model.fnn(x_u_ph, W_u, b_u)
    f_pred = model.pdenn(x_f_ph, W_u, b_u)

    #loss for bcs
    #loss_u = tf.reduce_mean(tf.square(u_pred - u_ph))
    #loss for RHS
    loss_f = tf.reduce_mean(tf.square(f_pred - f_ph))

    #loss = loss_u + loss_f
    loss = loss_f

    train = tf.train.AdamOptimizer(learning_rate=1.0e-3).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    n = 0
    nmax = 20000
    train_dict = {x_u_ph: x_u_train, u_ph: u_train, x_f_ph: x_f_train, f_ph: f_train}

    start_time = time.perf_counter()
    while n <= nmax:
        loss_, _ = sess.run([loss, train], feed_dict=train_dict)
        n += 1
        if n%1000 == 0:
            print('Steps: %d, Loss: %.3e'%(n, loss_))
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))

    num_test = 1001
    x_test = np.linspace(-1., 1., num_test).reshape(-1, 1)
    u_test, f_test = sess.run([u_pred, f_pred], feed_dict={x_u_ph: x_test, x_f_ph: x_test})
    save_dict = {'x_u_train': x_u_train, 'u_train': u_train, 'x_f_train': x_f_train, 'f_train': f_train, 'x_test': x_test, \
                 'u_test': u_test, 'f_test': f_test}
    io.savemat('./Output/ode_pred.mat', save_dict)

    u_ref, f_ref = np.sin(np.pi*x_test), np.pi**2*np.sin(np.pi*x_test)

    err_u, err_f = np.linalg.norm(u_test - u_ref)/np.linalg.norm(u_ref), np.linalg.norm(f_test - f_ref)/np.linalg.norm(f_ref)
    print('Relative l2 errors: u: %3e, f: %.3e'%(err_u, err_f))

    plt.figure()
    plt.plot(x_u_train, u_train, 'bo')
    plt.plot(x_test, u_ref, 'k-')
    plt.plot(x_test, u_test, 'r--')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.show()

    plt.figure()
    plt.plot(x_f_train, f_train, 'bo')
    plt.plot(x_test, f_ref, 'k-')
    plt.plot(x_test, f_test, 'r--')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.show()
     

if __name__ == '__main__':
    main()
