import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.io as io
import time

from net import DNN

np.random.seed(1234)
tf.set_random_seed(1234)

tfd = tfp.distributions

num_samples = int(1e2)
num_burnin = int(2e3)

layers = [1] + [50]*2 + [1]
noise_u = 0.01
noise_f = 0.05

def load_data():
    data = io.loadmat("./Data/ode_data.mat")
    t, u, f = data["t"], data["y"], data["f"]
    t = t.reshape((-1, 1)); u = u.reshape((-1, 1)); f = f.reshape((-1, 1))

    NE = t.shape[0]

    num_u = 50
    u_idx = np.random.choice(NE, num_u, replace=False)
    t_u_train = t[u_idx]; u_train = u[u_idx]
    u_train = u_train + noise_u*np.random.normal(0, 1, size=u_train.shape)

    num_f = 50
    f_idx = np.random.choice(NE, num_f, replace=False)
    t_f_train = t[f_idx]; f_train = f[f_idx]
    f_train = f_train + noise_f*np.random.normal(0, 1, size=f_train.shape)

    return t_u_train, u_train, t_f_train, f_train, t, u, f

def make_prior():
    prior_w_0 = tfd.Normal(loc=0, scale=1)
    prior_w_1 = tfd.Normal(loc=0, scale=1)
    prior_w_2 = tfd.Normal(loc=0, scale=1)
    prior_b_0 = tfd.Normal(loc=0, scale=1)
    prior_b_1 = tfd.Normal(loc=0, scale=1)
    prior_b_2 = tfd.Normal(loc=0, scale=1)
    prior_a = tfd.Normal(loc=0, scale=1)
    return prior_w_0, prior_w_1, prior_w_2, prior_b_0, prior_b_1, prior_b_2, prior_a

def make_likelihood(t_u, t_f, model, w_u_0, w_u_1, w_u_2, b_u_0, b_u_1, b_u_2, a):
    #surrogate model
    W_u = [w_u_0, w_u_1, w_u_2]; b_u = [b_u_0, b_u_1, b_u_2]
    _, u_pred = model.fnn(t_u, W_u, b_u)
    ll_u = tfd.Normal(loc=u_pred, scale = noise_u*tf.ones_like(u_pred))
    f_pred = model.pde(t_f, W_u, b_u, a)
    ll_f = tfd.Normal(loc=f_pred, scale = noise_f*tf.ones_like(f_pred))
    return ll_u, ll_f


def main():
    t_u_train, u_train, t_f_train, f_train, t_test, u_test, f_test = load_data()

    t_u_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    u_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    t_f_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    f_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    u_train_tf = tf.convert_to_tensor(u_train, dtype=tf.float32)
    f_train_tf = tf.convert_to_tensor(f_train, dtype=tf.float32)

    model = DNN()

    #initializations for HMC
    w_u_0_init = tf.zeros([layers[0], layers[1]], dtype=tf.float32)
    w_u_1_init = tf.zeros([layers[1], layers[2]], dtype=tf.float32)
    w_u_2_init = tf.zeros([layers[2], layers[3]], dtype=tf.float32)
    b_u_0_init = tf.zeros([1, layers[1]], dtype=tf.float32)
    b_u_1_init = tf.zeros([1, layers[2]], dtype=tf.float32)
    b_u_2_init = tf.zeros([1, layers[3]], dtype=tf.float32)
    a_init = tf.zeros([1, 1], dtype=tf.float32)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def mcmc_sample():
        prior_w_u_0, prior_w_u_1, prior_w_u_2, prior_b_u_0, prior_b_u_1, prior_b_u_2, prior_a = make_prior()
        def posterior(w_u_0, w_u_1, w_u_2, b_u_0, b_u_1, b_u_2, a):
            ll_u, ll_f = make_likelihood(t_u_train, t_f_train, model, w_u_0, w_u_1, w_u_2, b_u_0, b_u_1, b_u_2, a)
            return (tf.reduce_sum(prior_w_u_0.log_prob(w_u_0)) + \
                    tf.reduce_sum(prior_w_u_1.log_prob(w_u_1)) + \
                    tf.reduce_sum(prior_w_u_2.log_prob(w_u_2)) + \
                    tf.reduce_sum(prior_b_u_0.log_prob(b_u_0)) + \
                    tf.reduce_sum(prior_b_u_1.log_prob(b_u_1)) + \
                    tf.reduce_sum(prior_b_u_2.log_prob(b_u_2)) + \
                    tf.reduce_sum(prior_a.log_prob(a)) + \
                    tf.reduce_sum(ll_u.log_prob(u_train_tf)) + \
                    tf.reduce_sum(ll_f.log_prob(f_train_tf)))

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
                                   current_state = [w_u_0_init, w_u_1_init, w_u_2_init, b_u_0_init, b_u_1_init, b_u_2_init, a_init],
                                   kernel = adaptive_hmc,
                                   trace_fn = lambda _, pkr: pkr.inner_results.is_accepted)
            return samples, is_accepted

        samples_, is_accepted_ = run_chain()
        return samples_, is_accepted_

    #run HMC
    start_time = time.perf_counter()
    samples, is_accepted = mcmc_sample()
    train_dict = {t_u_ph: t_u_train, u_ph: u_train, t_f_ph: t_f_train, f_ph: f_train}
    samples_, is_accepted_ = sess.run([samples, is_accepted], feed_dict=train_dict)
    w0_u_samples, w1_u_samples, w2_u_samples, b0_u_samples, b1_u_samples, b2_u_samples, a_samples = samples
    stop_time = time.perf_counter()
    print('Duration time is %.3f seconds'%(stop_time - start_time))

    W_u_samples = [w0_u_samples, w1_u_samples, w2_u_samples]; b_u_samples = [b0_u_samples, b1_u_samples, b2_u_samples]
    is_accepted_test = is_accepted

    acceptance_rate = np.mean(is_accepted_)
    print('Acceptance rate: %.3f'%(acceptance_rate))

    print('Saving data ...')

    X_test = tf.convert_to_tensor(t_test, dtype=tf.float32)
    
    _, u_out = model.fnnout(X_test, W_u_samples, b_u_samples)
    f_out = model.pdeout(X_test, W_u_samples, b_u_samples, a_samples)
    u_pred_, a_pred_, f_pred_ = sess.run([u_out, a_samples, f_out])

    save_dict = {'t_u_train': t_u_train, 'u_train': u_train, 't_f_train': t_f_train, 'f_train': f_train, 't_test': t_test, 'u_test': u_test, 'f_test': f_test, \
                'u_pred': u_pred_, 'a_pred': a_pred_, 'f_pred': f_pred_}
    io.savemat('./Output/ode_pred.mat', save_dict)


if __name__ == '__main__':
    main()
