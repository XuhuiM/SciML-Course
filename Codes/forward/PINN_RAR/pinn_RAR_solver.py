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
layers = [1] + 2*[30] + [1]


def exact_u_sol(x):
    u = x + np.sin(x) + np.sin(2*x)/2 + np.sin(3*x)/3 + np.sin(4*x)/4 + np.sin(8*x)/8
    return u

def exact_f_sol(x):
    f = np.sin(x) + 2*np.sin(2*x) + 3*np.sin(3*x) + 4*np.sin(4*x) + 8*np.sin(8*x)
    return f

def training_data():
    xmin, xmax = -np.pi, np.pi
    num_f_train = 20
    x_f_train = np.linspace(xmin, xmax, num_f_train).reshape((-1, 1))
    f_train = exact_f_sol(x_f_train)
    x_bc_l = np.array([xmin]).reshape((-1, 1))
    x_bc_r = np.array([xmax]).reshape((-1, 1))
    u_l, u_r = exact_u_sol(x_bc_l), exact_u_sol(x_bc_r)
    x_u_train = np.vstack((x_bc_l, x_bc_r))
    u_train = np.vstack((u_l, u_r))

    return x_u_train, u_train, x_f_train, f_train

def add_data(x_r_test, err_current):
    x_id = np.argmax(np.absolute(err_current))
    x_add = x_r_test[x_id]
    return x_add

def build_dataset(x_current, x_add):
    x_add = np.reshape(x_add, (-1, 1))
    x_f_batch = np.vstack((x_current, x_add))
    return x_f_batch

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
    f_pred, r_pred = model.pdenn(x_f_ph, W_u, b_u)

    #loss for bcs
    loss_u = tf.reduce_mean(tf.square(u_pred - u_ph))
    #loss for residual
    loss_r = tf.reduce_mean(tf.square(r_pred))

    loss = loss_u + loss_r

    train = tf.train.AdamOptimizer(learning_rate=1.0e-3).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    n = 0
    nmax = 30
    err = 1.

    #loop for residual
    n_r_max = 20000

    start_time = time.perf_counter()
    while n <= nmax and err > 1.0e-2:
        print('Loop: %d'%(n+1))
        if n == 0:
            x_f_batch = x_f_train
        else:
            x_f_batch = build_dataset(x_f_batch, x_add)
        
        n_r = 0
        loss_r_ = 1.0
        while n_r <= n_r_max and loss_r_ >= 1.0e-6: 
            train_dict = {x_u_ph: x_u_train, u_ph: u_train, x_f_ph: x_f_batch}
            loss_r_, _ = sess.run([loss, train], feed_dict=train_dict)
            n_r += 1
            if n_r%100 == 0:
                print('Steps: %d, Loss: %.3e'%(n_r, loss_r_))
        n += 1
        
        print('# of training data: %d'%(x_f_batch.shape[0]))
        x_r_test = 2*np.pi*np.random.rand(100, 1) - np.pi
        f_test, r_test = sess.run([f_pred, r_pred], feed_dict = {x_f_ph: x_r_test})
        err = np.mean(np.absolute(r_test))
        print('Mean error of residual: %.5f'%(err))
        x_add = add_data(x_r_test, r_test)
        print('Added points is %.3f'%(x_add))


    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))

    f_train_batch = sess.run(f_pred, feed_dict={x_f_ph: x_f_batch})
    num_test = 1001
    x_test = np.linspace(-np.pi, np.pi, num_test).reshape(-1, 1)
    u_test, f_test = sess.run([u_pred, f_pred], feed_dict={x_u_ph: x_test, x_f_ph: x_test})
    save_dict = {'x_u_train': x_u_train, 'u_train': u_train, 'x_f_train': x_f_train, 'f_train': f_train, 'x_test': x_test, \
                 'u_test': u_test, 'f_test': f_test}
    io.savemat('./Output/ode_pred.mat', save_dict)

    u_ref, f_ref = exact_u_sol(x_test), exact_f_sol(x_test)

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
    plt.plot(x_f_batch, f_train_batch, 'b*')
    plt.plot(x_test, f_ref, 'k-')
    plt.plot(x_test, f_test, 'r--')
    plt.xlabel('x')
    plt.ylabel('f')
    plt.show()
     

if __name__ == '__main__':
    main()
