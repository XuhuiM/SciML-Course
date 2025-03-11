import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

from net import DNN 

tf.disable_v2_behavior()

layers = [1] + [500]

np.random.seed(1234)
tf.set_random_seed(1234)

def main():
    num = 1001
    x = np.linspace(-1, 1, num).reshape((-1, 1))
    y = np.sin(np.pi*x)
    R = 1.

    num_col = layers[1]
    x_col = np.linspace(-1.1, 1.1, num_col).reshape((1, -1))

    Xmin = x.min(0)
    Xmax = x.max(0)

    x_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    model = DNN()
    W, b = model.hyper_initial(layers, R, x_col)

    basis, feature = model.fnn(x_train, W, b, Xmin, Xmax)
    #w = tf.linalg.lstsq(basis, y_train, l2_regularizer=1.0e-6)

    '''
    y_pred = model.fnn(x_train, W, b, Xmin, Xmax)

    loss = tf.reduce_mean(tf.square(y_pred - y_train))
    train = tf.train.AdamOptimizer().minimize(loss)
    '''

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #c = sess.run(w, feed_dict={x_train: x, y_train: y})
    basis_train = sess.run(basis, feed_dict={x_train: x})
    feature_train = sess.run(feature[0], feed_dict={x_train: x})
    '''
    a = np.matmul(np.transpose(basis_train), basis_train)
    a = a + 1.0e-3*np.identity(a.shape[0])
    b = np.matmul(np.transpose(basis_train), y)
    w = np.linalg.solve(a, b)
    '''
    w, _, _, _ = np.linalg.lstsq(basis_train, y)

    '''
    plt.figure()
    plt.subplot(1,2, 1)
    plt.plot(x, feature_train)
    plt.subplot(1,2, 2)
    plt.plot(x, basis_train)
    plt.show()
    '''

    '''
    nmax = 10000
    n = 0
    while n < nmax:
        n += 1
        y_, loss_, _ = sess.run([y_pred, loss, train], feed_dict={x_train: x, y_train: y})
        if n%100 == 0:
            print('n: %d, loss: %.3e'%(n, loss_))
    '''
    
    num_test = 2001
    x_test = np.linspace(-1, 1, num_test).reshape((-1, 1))
    basis_test = sess.run(basis, feed_dict={x_train: x_test})
    y_test = np.matmul(basis_test, w)
    #y_test = sess.run(y_pred, feed_dict={x_train: x_test})
    y_ref = np.sin(np.pi*x_test)

    err = np.linalg.norm(y_test - y_ref)/np.linalg.norm(y_ref)
    print('Error(L2): %.5e'%(err))

    plt.figure()
    plt.plot(x, y, 'b.')
    plt.plot(x_test, y_ref, 'k-')
    plt.plot(x_test, y_test, 'r--')
    plt.show()

if __name__ == '__main__':
    main()
