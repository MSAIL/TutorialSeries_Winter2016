'''
Thanks to Edwin Chen!
Code to calculate clusters using a Dirichlet Process
Gaussian mixture model.
Requires scikit-learn:
  http://scikit-learn.org/stable/
'''

import numpy
from sklearn import mixture

import itertools
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

FILENAME = "mcdonalds-normalized-data-clean.tsv"

# Note: you'll have to remove the last "name" column in the file (or
# some other such thing), so that all the columns are numeric.
X = numpy.loadtxt(open(FILENAME, "rb"), delimiter = "\t", skiprows = 1)
dpgmm = mixture.DPGMM(n_components = 25)
dpgmm.fit(X)
clusters = dpgmm.predict(X)

classes = [[] for i in range(25)]
for i,c in enumerate(clusters):
    classes[c].append(i)

with open('mcdonalds-normalized-data-names.tsv') as f:
    names = f.read().split('\n')[1:]
with open('chen_out','w') as f:
    f.write('\n\n\n'.join('\n'.join(names[i] for i in cc) for cc in classes))

for i, (clf, title) in enumerate([(dpgmm, 'Dirichlet Process GMM')]):
    splot = plt.subplot(1, 1, 1 + i)
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
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-1, 10)
    plt.ylim(-1, 10)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

plt.show()
