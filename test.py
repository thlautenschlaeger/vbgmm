from gmm import VBGMM
import numpy as np
from plot_lib import plot_2d_gaussian

import matplotlib.pyplot as plt

faithful = np.loadtxt('faithful.csv', usecols=[1, 2], delimiter=',', skiprows=1)

# set hyper parameters
k = 6
alpha = np.ones(k) / 100
beta = 0.05
m = 0
W_0 = np.eye(2) * 0.5

nu = 10

vbgmm = VBGMM(x=faithful, k=k, alpha_0=alpha, beta_0=beta, m_0=m, W_0=W_0, nu_0=nu, normalize=True)

elbos = []
for i in range(30):
    vbgmm.optimize()
    elbo = vbgmm.compute_vlb()

    elbos.append(elbo.squeeze(axis=1))

    if i % 10 == 0:
        plot_2d_gaussian(vbgmm.x, vbgmm.means, vbgmm.covars, 100)
        plt.scatter(vbgmm.x[:, 0], vbgmm.x[:, 1])
        plt.show()
    pass

plt.plot(elbos)
plt.show()

# plot_2d_gaussian(vbgmm.x, vbgmm.means, vbgmm.covars, 100)
# plt.scatter(vbgmm.x[:, 0], vbgmm.x[:, 1])
# plt.show()

