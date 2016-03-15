import itertools
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

X = np.genfromtxt('xdata.csv', delimiter=',', names=['x', 'y'])
X = np.array([list(x) for x in X]) #numpy likes lists

# Fit a mixture of Gaussians with EM using five components
gmm = mixture.GMM(n_components=2, covariance_type='full')
gmm.fit(X)

# Fit a Dirichlet process mixture of Gaussians using no more than 100=infinity components
dpgmm = mixture.DPGMM(alpha=0.1, n_components=100, covariance_type='full')
dpgmm.fit(X)

for i, (clf, title) in enumerate([(gmm, 'GMM'),
                                  (dpgmm, 'Dirichlet Process GMM')]):
    splot = plt.subplot(2, 1, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0]*2, v[1]*2, 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

plt.show()
