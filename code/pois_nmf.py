"""
Poisson GaP-NMF

CREATED: 2013-08-09 23:50:58 by Dawen Liang <dl2771@columbia.edu>

"""

import functools

import numpy as np
import matplotlib.pyplot as plt

import gap_nmf
import _gap

specshow = functools.partial(plt.imshow, cmap=plt.cm.jet, origin='lower',
                             aspect='auto', interpolation='nearest')


class GaP_NMF (gap_nmf.GaP_NMF):
    def __init__(self, X, K=100, smoothness=100, seed=None, **kwargs):
        super(GaP_NMF, self).__init__(X, K=K, smoothness=smoothness,
                                         seed=seed, **kwargs)

    def update_w(self):
        goodk = self.goodk()
        pass

    def update_h(self):
        goodk = self.goodk()
        pass

    def update_theta(self):
        goodk = self.goodk()
        pass

    def bound(self):
        pass
