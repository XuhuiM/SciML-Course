import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

from net import DNN 

layers = [1] + 3*[20] + [1]

np.random.seed(1234)
#tf.set_random_seed(1234)

def main():
    x = np.linspace(-1, 1, 21).reshape((-1, 1))
    y = x**2

    Xmin = x.min(0)
    Xmax = x.max(0)

    x_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_train = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    model = DNN(layers)
    W, b = model.hyper_initial()
    y_pred = model.fnn(x_train, W, b, Xmin, Xmax)

    loss = tf.reduce_mean(tf.square(y_pred - y_train))
    train = tf.train.AdamOptimizer().minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nmax = 10000
    n = 0
    while n < nmax:
        n += 1
        y_, loss_, _ = sess.run([y_pred, loss, train], feed_dict={x_train: x, y_train: y})
        if n%100 == 0:
            print('n: %d, loss: %.3e'%(n, loss_))
    
    x_test = np.linspace(-1, 1, 1001).reshape((-1, 1))
    y_ref = x_test**2
    y_test = sess.run(y_pred, feed_dict={x_train: x_test})
    plt.figure()
    plt.plot(x, y, 'bo')
    plt.plot(x_test, y_ref, 'k-')
    plt.plot(x_test, y_test, 'r--')
    plt.show()

if __name__ == '__main__':
    main()
