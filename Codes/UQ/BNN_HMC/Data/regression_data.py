import numpy as np
import scipy.io as io

noise = 0.1

np.random.seed(1234)

def main():
    num = 16
    x1 = np.linspace(-0.8, -0.2, num).reshape((-1, 1))
    x2 = np.linspace( 0.2,  0.8, num).reshape((-1, 1))
    x = np.vstack((x1, x2))
    u = np.sin(2*np.pi*x)**3 + noise*np.random.normal(0, 1, size=x.shape)

    np.savez('Regression_Data', u_x=x, u=u)
    save_dict = {'u_x': x, 'u': u}
    io.savemat('Regression_Data.mat', save_dict)


if __name__ == '__main__':
    main()
