import numpy as np
import scipy.io as io

class TrainData:
    def __init__(self, t_range, NT_train, N_train):
        self.t_range = t_range
        self.NT = NT_train
        self.N_train = N_train

    def build_data(self):
        data = io.loadmat('./Data/VIV_Training')
        t, u, f = data['t'], data['eta_y'], data['f_y']

        N = 160
        t_u_idx = np.random.choice(N, self.NT, replace=False)
        t_u = t[t_u_idx]
        u_data = u[t_u_idx]
        
        t_f_idx = np.random.choice(N, self.N_train, replace=False)
        t_f = t[t_f_idx]
        f_data = f[t_f_idx]

        t_ref, u_ref, f_ref = t[:N, :], u[:N, :], f[:N, :]
        return t_u, u_data, t_f, f_data, t_ref, u_ref, f_ref
