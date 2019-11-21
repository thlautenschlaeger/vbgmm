import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_2d_gaussian(data, mean, covar, bins):
    """ Plot 2d Gaussian pdf """
    x = data[:, 0]
    y = data[:, 1]

    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10

    xmin = min(x) - deltaX
    xmax = max(x) + deltaX

    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    xx = np.linspace(xmin, xmax, bins)
    yy = np.linspace(ymin, ymax, bins)

    X, Y = np.meshgrid(xx, yy)

    if len(mean.shape) > 2:
        for k in range(mean.shape[0]):
            Z = multivariate_normal.pdf(np.dstack((X, Y)), mean[k].squeeze(), covar[k], allow_singular=True)
            plt.contour(X, Y, Z)
    else:
        Z = multivariate_normal.pdf(np.dstack((X, Y)), mean.squeeze(), covar)
        plt.contour(X, Y, Z)