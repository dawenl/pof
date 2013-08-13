"""
Poisson GaP-NMF

CREATED: 2013-08-09 23:50:58 by Dawen Liang <dl2771@columbia.edu>

"""

import functools

import numpy as np
import matplotlib.pyplot as plt

import _gap

specshow = functools.partial(plt.imshow, cmap=plt.cm.jet, origin='lower',
                             aspect='auto', interpolation='nearest')


class GaP_NMF:
    def __init__(self, X, K=100, smoothness=100, seed=None, **kwargs):
        self.X = X / np.mean(X)
        self.K = K
        self.F, self.T = X.shape
        if seed is None:
            print 'Using random seed'
            np.random.seed()
        else:
            print 'Using fixed seed {}'.format(seed)
            np.random.seed(seed)
        self._parse_args(**kwargs)
        self._init(smoothness)

    def _parse_args(self, **kwargs):
        self.a = float(kwargs['a']) if 'a' in kwargs else 0.1
        self.b = float(kwargs['b']) if 'b' in kwargs else 0.1
        self.alpha = float(kwargs['alpha']) if 'alpha' in kwargs else 1.
        self.d = int(kwargs['scale']) if 'scale' in kwargs else 100

    def _init(self, smoothness):
        self.nuw = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                           size=(self.F, self.K))
        self.rhow = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                            size=(self.F, self.K))
        self.nuh = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                           size=(self.F, self.K))
        self.rhoh = 10000 * np.random.gamma(smoothness, 1./smoothness,
                                            size=(self.K, self.T))
        self.nut = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                           size=(self.K, ))
        self.rhot = self.K * 10000 * np.random.gamma(smoothness, 1./smoothness,
                                                     size=(self.K, ))
        self.compute_expectations()

    def compute_expectations(self):
        self.Ew, self.Elogw = _gap.compute_gamma_expectation(self.nuw,
                                                             self.rhow)
        self.Eh, self.Elogh = _gap.compute_gamma_expectation(self.nuh,
                                                             self.rhoh)
        self.Et, self.Elogt = _gap.compute_gamma_expectation(self.nut,
                                                             self.rhot)

    def update(self):
        self.update_h()
        self.update_w()
        self.update_theta()
        # truncate unused components
        self.clear_badk()

    def update_w(self):
        goodk = self.goodk()
        pass

    def update_h(self):
        goodk = self.goodk()
        pass

    def update_theta(self):
        goodk = self.goodk()
        pass

    def goodk(self):
        pass

    def clear_badk(self):
        pass

    def bound(self):
        pass
