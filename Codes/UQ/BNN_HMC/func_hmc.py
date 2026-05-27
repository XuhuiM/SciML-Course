import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import time

from net import DNN

np.random.seed(1234)
'''
tf.random.set_seed(1234)
'''
tf.set_random_seed(1234)

tfd = tfp.distributions

num_samples = int(1e3)
num_burnin = int(2e3)

layers = [1] + [50]*2 + [1]
noise_scale = 0.1

def generate_training_data():
    data = io.loadmat('./Data/Regression_Data.mat')
    x = data['u_x']
    y = data['u']
    x = np.reshape(x, (-1, 1))
    x = x.astype(np.float32)
    y = np.reshape(y, (-1, 1))
    y = y.astype(np.float32)

    num_test = 101
    x_vld = np.linspace(-1, 1, num_test).reshape((-1, 1))
    y_vld = np.sin(2*np.pi*x_vld)**3

    return x, y, x_vld, y_vld

def make_prior():
    prior_w_0 = tfd.Normal(loc=0, scale=1)
    prior_w_1 = tfd.Normal(loc=0, scale=1)
    prior_w_2 = tfd.Normal(loc=0, scale=1)
    prior_b_0 = tfd.Normal(loc=0, scale=1)
    prior_b_1 = tfd.Normal(loc=0, scale=1)
    prior_b_2 = tfd.Normal(loc=0, scale=1)
    return prior_w_0, prior_w_1, prior_w_2, prior_b_0, prior_b_1, prior_b_2 

def make_likelihood(X, w_0, w_1, w_2, b_0, b_1, b_2):
    #surrogate model
#    y_pred = tf.add(tf.matmul(X, w), b)
    dnn_model = DNN(layers, w_0, w_1, w_2, b_0, b_1, b_2)
    y_pred = dnn_model.fnn(X)
    return tfd.Normal(loc=y_pred, scale = noise_scale*tf.ones_like(y_pred))


def main():
    x_train, y_train, x_vld, y_vld = generate_training_data()

    x_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    y_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    w_0_init = tf.zeros([layers[0], layers[1]], dtype=tf.float32)
    w_1_init = tf.zeros([layers[1], layers[2]], dtype=tf.float32)
    w_2_init = tf.zeros([layers[2], layers[3]], dtype=tf.float32)
    b_0_init = tf.zeros([1, layers[1]], dtype=tf.float32)
    b_1_init = tf.zeros([1, layers[2]], dtype=tf.float32)
    b_2_init = tf.zeros([1, layers[3]], dtype=tf.float32)

    def mcmc_sample():
        prior_w_0, prior_w_1, prior_w_2, prior_b_0, prior_b_1, prior_b_2 = make_prior()
        def posterior(w_0, w_1, w_2, b_0, b_1, b_2):
            likelihood = make_likelihood(x_ph, w_0, w_1, w_2, b_0, b_1, b_2)
            return (tf.reduce_sum(prior_w_0.log_prob(w_0)) + \
                    tf.reduce_sum(prior_w_1.log_prob(w_1)) + \
                    tf.reduce_sum(prior_w_2.log_prob(w_2)) + \
                    tf.reduce_sum(prior_b_0.log_prob(b_0)) + \
                    tf.reduce_sum(prior_b_1.log_prob(b_1)) + \
                    tf.reduce_sum(prior_b_2.log_prob(b_2)) + \
                    tf.reduce_sum(likelihood.log_prob(y_ph)))

        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
                       tfp.mcmc.HamiltonianMonteCarlo(
                       target_log_prob_fn=posterior,
                       num_leapfrog_steps=50,
                       step_size=0.1),
                       num_adaptation_steps = int(num_burnin*0.8))

        def run_chain():
            samples, is_accepted = tfp.mcmc.sample_chain(
                                   num_results = num_samples,
                                   num_burnin_steps = num_burnin,
                                   current_state = [w_0_init, w_1_init, w_2_init, b_0_init, b_1_init, b_2_init],
                                   kernel = adaptive_hmc,
                                   trace_fn = lambda _, pkr: pkr.inner_results.is_accepted)
            return samples, is_accepted

        samples_, is_accepted_ = run_chain()
        return samples_, is_accepted_

    start_time = time.perf_counter()
    samples, is_accepted = mcmc_sample()
    train_dict = {x_ph: x_train, y_ph: y_train}
    samples_, is_accepted_ = sess.run([samples, is_accepted], feed_dict=train_dict)
    w0_samples, w1_samples, w2_samples, b0_samples, b1_samples, b2_samples = samples_
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))

    w0_samples_test = tf.constant(w0_samples)
    w1_samples_test = tf.constant(w1_samples)
    w2_samples_test = tf.constant(w2_samples)
    b0_samples_test = tf.constant(b0_samples)
    b1_samples_test = tf.constant(b1_samples)
    b2_samples_test = tf.constant(b2_samples)
    is_accepted_test = is_accepted

    acceptance_rate = np.mean(is_accepted_)
    print('Acceptance rate: %.3f'%(acceptance_rate))

    print('Saving data ...')

    X = x_vld
    X_test = tf.constant(X, dtype=tf.float32)
    
    model_output = DNN(layers, w0_samples_test, w1_samples_test, w2_samples_test, b0_samples_test, b1_samples_test, b2_samples_test)
    Y_test = model_output.fnn_output(X_test)
    Y_test = sess.run(Y_test)

    Y_test = Y_test[:, :, 0]

    y_mean = np.mean(Y_test, axis=0)
    y_std = np.std(Y_test, axis=0)
    y_lb, y_ub = y_mean - 2*y_std, y_mean + 2*y_std


    plt.plot(x_vld, y_vld, 'k-')
    plt.plot(x_train, y_train, 'bo')
    plt.plot(x_vld, y_mean, 'r--')
    plt.fill_between(x_vld.ravel(), y_lb, y_ub, facecolor='c', alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()
