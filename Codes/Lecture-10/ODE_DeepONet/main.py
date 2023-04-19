import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as io

from dataset import DataSet
from net import DNN

np.random.seed(1234)
tf.set_random_seed(1234)

x_dim = 1
x_num = 50
#input dimension for Branch Net
f_dim = x_num
#output dimension for Branch and Trunk Net
G_dim = x_num

#Branch Net
layers_f = [f_dim] + [50]*2 + [G_dim]
#Trunk Net
layers_x = [x_dim] + [50]*2 + [G_dim]

batch_size = 128
num_test = 1000

def main():
    data = DataSet(x_num, batch_size)

    x_train, f_train, u_train = data.minibatch()

    x_pos = tf.constant(x_train, dtype=tf.float32) #[x_num, x_dim]
    x = tf.tile(x_pos[None, :, :], [batch_size, 1, 1]) #[batch_size, x_num, x_dim]
    X_test = tf.tile(x_pos[None, :, :], [num_test, 1, 1]) #[num_test, x_num, x_dim]
    Xmin = -1
    Xmax = 1

    #placeholder for f
    f_ph = tf.placeholder(shape=[None, f_dim], dtype=tf.float32)#[bs, f_dim]
    u_ph = tf.placeholder(shape=[None, x_num], dtype=tf.float32)#[bs, x_num]

    #model
    model = DNN()

    W_g_f, b_g_f = model.hyper_initial(layers_f)
    W_g_x, b_g_x = model.hyper_initial(layers_x)
    u_f = model.fnn_B(f_ph, W_g_f, b_g_f)
    u_f = tf.tile(u_f[:, None, :], [1, x_num, 1]) #[batch_size, x_num, G_dim]
    u_x = model.fnn_T(x, W_g_x, b_g_x, Xmin, Xmax) #[batch_size, x_num, G_dim]
    u_pred = u_f*u_x
    u_pred = tf.reduce_sum(u_pred, axis=-1)
    
    var_list = [W_g_f, b_g_f, W_g_x, b_g_x]

	#loss
    loss = tf.reduce_mean(tf.square(u_ph - u_pred))
    train = tf.train.AdamOptimizer(learning_rate=1.0e-3).minimize(loss, var_list=var_list)
    
    #save model
    saver = tf.train.Saver([weight for weight in W_g_f+b_g_f+W_g_x+b_g_x])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    n = 0
    nmax = 50000
    start_time = time.perf_counter()
    time_step_0 = time.perf_counter()
    while n <= nmax:
        x_train, f_train, u_train = data.minibatch()
        train_dict = {f_ph: f_train, u_ph: u_train}
        loss_, _ = sess.run([loss, train], feed_dict=train_dict)

        if n%100 == 0:
            _, _, f_test, u_test = data.testbatch(batch_size)
            test_dict = {f_ph: f_test, u_ph: u_test}
            u_test_pred = sess.run(u_pred, feed_dict=test_dict)
            u_test_pred = data.decode(u_test_pred)
            err = np.mean(np.linalg.norm(u_test - u_test_pred, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))

            time_step_1000 = time.perf_counter()
            T = time_step_1000 - time_step_0
            print('Step: %d, loss: %.3e, Test L2 error: %.3e, Time: %.3f'%(n, loss_, err, T))
            time_step_0 = time.perf_counter()

        if n%10000 == 0:
            filename = './checkpoint/prior_' + str(n)
            saver.save(sess, filename)
        n += 1

    saver.save(sess, './checkpoint/model')

    test_id, x_test, f_test, u_test = data.testbatch(num_test)
    test_dict = {f_ph: f_test, u_ph: u_test}
    u_x_test = model.fnn_T(x_test, W_g_x, b_g_x, Xmin, Xmax) #[num_test, x_num, G_dim]
    u_pred_test = u_f*u_x_test
    u_pred_test = tf.reduce_sum(u_pred_test, axis=-1)
    u_pred_test_ = sess.run(u_pred_test, feed_dict=test_dict)
    u_pred_test_ = data.decode(u_pred_test_)
    err = np.mean(np.linalg.norm(u_test - u_pred_test_, 2, axis=1)/np.linalg.norm(u_test, 2, axis=1))
    print('L2 error: %.3e'%err)
    save_dict = {'test_id': test_id, 'x_test': x_test, 'f_test': f_test, 'u_test': u_test, 'u_pred': u_pred_test_, 'l2': err}
    io.savemat('./Output/ODE_Preds.mat', save_dict)

    end_time = time.perf_counter()
    print('Elapsed time: %.3f seconds'%(end_time - start_time))

if __name__ == '__main__':
    main()
