"""

IS-NMF

CREATED: 2013-08-08 13:43:27 by Dawen Liang <daliang@adobe.com>

"""

import functools
from math import log
import time

import numpy as np
import matplotlib.pyplot as plt

import _gap


specshow = functools.partial(plt.imshow, cmap=plt.cm.jet, origin='lower',
                             aspect='auto', interpolation='nearest')


class IS_NMF:
    def __init__(self, X, K=100, smoothness=100, seed=None, **kwargs):
        self.X = X.copy()
        self.K = K
        self.F, self.T = X.shape

        if seed is None:
            print 'Using random seed'
        else:
            print 'Using fixed seed {}'.format(seed)

        self._parse_args(**kwargs)
        self._init(smoothness)

    def _parse_args(self, **kwargs):
        self.a = kwargs['a'] if 'a' in kwargs else 0.1
        self.b = kwargs['b'] if 'b' in kwargs else 0.1

    def _init(self, smoothness):
        self.rhow = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.F, self.K))
        self.tauw = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.F, self.K))
        self.rhoh = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.K, self.T))
        self.tauh = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.K, self.T))
        self.compute_expectations()

    def compute_expectations(self):
        self.Ew, self.Ewinv = _gap.compute_gig_expectations(self.a,
                                                            self.rhow,
                                                            self.tauw)
        self.Ewinvinv = 1. / self.Ewinv
        self.Eh, self.Ehinv = _gap.compute_gig_expectations(self.b,
                                                            self.rhoh,
                                                            self.tauh)
        self.Ehinvinv = 1. / self.Ehinv

    def update(self, disp=0):
        ''' Do optimization for one iteration
        '''
        self.update_h()
        self.update_w()

    def update_w(self):
        xtwid = self._xtwid()
        c = np.mean(self.X / xtwid)
        print('Optimal scale for updating W: {}'.format(c))
        xxtwidinvsq = self.X / c * xtwid**(-2)
        xbarinv = 1. / self._xbar()

        self.rhow = self.a + np.dot(xbarinv, self.Eh.T)
        self.tauw = self.Ewinvinv**2 * np.dot(xxtwidinvsq, self.Ehinvinv.T)
        self.tauw[self.tauw < 1e-100] = 0

        self.Ew, self.Ewinv = _gap.compute_gig_expectations(self.a,
                                                            self.rhow,
                                                            self.tauw)
        self.Ewinvinv = 1. / self.Ewinv

    def update_h(self):
        xtwid = self._xtwid()
        c = np.mean(self.X / xtwid)
        print('Optimal scale for updating H: {}'.format(c))
        xxtwidinvsq = self.X / c * xtwid**(-2)
        xbarinv = 1. / self._xbar()
        self.rhoh = self.b + np.dot(self.Ew.T, xbarinv)
        self.tauh = self.Ehinvinv**2 * np.dot(self.Ewinvinv.T, xxtwidinvsq)
        self.tauh[self.tauh < 1e-100] = 0
        self.Eh, self.Ehinv = _gap.compute_gig_expectations(self.b,
                                                            self.rhoh,
                                                            self.tauh)
        self.Ehinvinv = 1. / self.Ehinv

    def bound(self):
        score = 0
        c = np.mean(self.X / self._xtwid())
        xbar = self._xbar()

        score = score - np.sum(np.log(xbar) + log(c))
        score = score + _gap.gig_gamma_term(self.Ew, self.Ewinv, self.rhow,
                                            self.tauw, self.a, self.a)
        score = score + _gap.gig_gamma_term(self.Eh, self.Ehinv, self.rhoh,
                                            self.tauh, self.b, self.b)
        return score

    def figures(self):
        ''' Animation-type of figures can only be created with PyGTK backend
        '''
        plt.subplot(2, 2, 1)
        specshow(np.log(self.Ew))
        plt.title('E[W]')
        plt.xlabel('component index')
        plt.ylabel('frequency')

        plt.subplot(2, 2, 2)
        specshow(np.log(self.Eh))
        plt.title('E[H]')
        plt.xlabel('time')
        plt.ylabel('component index')

        plt.subplot(2, 2, 3)
        specshow(np.log(self.X))
        plt.title('Original Spectrogram')
        plt.xlabel('time')
        plt.ylabel('frequency')

        plt.subplot(2, 2, 4)
        specshow(np.log(self._xbar()))
        plt.title('Reconstructed Spectrogram')
        plt.xlabel('time')
        plt.ylabel('frequency')

        time.sleep(0.000001)

    def _xbar(self):
        return np.dot(self.Ew, self.Eh)

    def _xtwid(self):
        return np.dot(self.Ewinvinv, self.Ehinvinv)
