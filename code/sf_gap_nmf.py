"""
Source-filter dictionary prior GaP-NMF

CREATED: 2013-07-25 15:09:04 by Dawen Liang <daliang@adobe.com> 

"""

import numpy as np
import scipy.optimize as optimize

import gap_nmf
import _gap

class SF_GaP_NMF(gap_nmf.GaP_NMF):
    def __init__(self, X, U, gamma, alpha, K=100, smoothness=100,
            seed=None, **kwargs):
        self.X = X / np.mean(X)
        self.K = K
        self.U = U.copy()
        self.alpha = alpha.copy()
        self.gamma = gamma.copy()
        
        self.F, self.T = X.shape
        self.L = alpha.size 

        if seed is None:
            print 'Using random seed'
        else:
            print 'Using fixed seed {}'.format(seed)

        self._parse_args(**kwargs)
        self._init(smoothness)

    def _parse_args(self, **kwargs):
        self.b = kwargs['b'] if 'b' in kwargs else 0.1
        self.beta = kwargs['beta'] if 'beta' in kwargs else 1.

    def _init(self, smoothness):
        self.rhow = 10000 * np.random.gamma(smoothness, 1./smoothness,
                size=(self.F, self.K))
        self.tauw = 10000 * np.random.gamma(smoothness, 1./smoothness,
                size=(self.F, self.K))
        self.rhoh = 10000 * np.random.gamma(smoothness, 1./smoothness,
                size=(self.K, self.T))
        self.tauh = 10000 * np.random.gamma(smoothness, 1./smoothness,
                size=(self.K, self.T))
        self.rhot = self.K * 10000 * np.random.gamma(smoothness, 1./smoothness,
                size=(self.K, ))
        self.taut = 1./self.K * 10000 * np.random.gamma(smoothness,
                1./smoothness, size=(self.K, ))
        self.gama = 10000 * np.random.gamma(smoothness, 1./smoothness,
                size=(self.L, self.K))
        self.rhoa = 10000 * np.random.gamma(smoothness, 1./smoothness,
                size=(self.L, self.K))
        self.compute_expectations()

    def compute_expectations(self):
        self.Ew, self.Ewinv = _gap.compute_gig_expectations(self.alpha, self.rhow,
                self.tauw)
        self.Ewinvinv = 1./self.Ewinv
        self.Eh, self.Ehinv = _gap.compute_gig_expectations(self.b, self.rhoh,
                self.tauh)
        self.Ehinvinv = 1./self.Ehinv
        self.Et, self.Etinv = _gap.compute_gig_expectations(self.beta/self.K,
                self.rhot, self.taut)
        self.Etinvinv = 1./self.Etinv


    def update(self):
        ''' Do optimization for one iteration
        '''
        self.update_h()
        self.update_a()
        self.update_w()
        self.update_theta()
        # truncate unused components
        self.clear_badk()


    def update_a(self):
        pass

    def update_w(self):
        goodk = self.goodk()
        xxtwidinvsq = (self.X * self._xtwid(goodk)) ** (-2)
        xbarinv = self._xbar(goodk) ** (-1)
        dEt = self.Et[goodk]
        dEtinvinv = self.Etinvinv[goodk]
        self.rhow[:, goodk] = self.a + np.dot(xbarinv, dEt * self.Eh[goodk,
            :].T)
        self.tauw[:, goodk] = self.Ewinvinv[:, goodk]**2 * \
                np.dot(xxtwidinvsq, dEtinvinv * self.Ehinvinv[goodk, :].T)
        self.tauw[self.tauw < 1e-100] = 0
        self.Ew[:, goodk], self.Ewinv[:, goodk] = _gap.compute_gig_expectations(
                self.a, self.rhow[:, goodk], self.tauw[:, goodk])
        self.Ewinvinv[:, goodk] = 1./self.Ewinv[:, goodk]

    def update_h(self):
        goodk = self.goodk()
        xxtwidinvsq = (self.X * self._xtwid(goodk)) ** (-2)
        xbarinv = self._xbar(goodk) ** (-1)
        dEt = self.Et[goodk]
        dEtinvinv = self.Etinvinv[goodk]
        self.rhoh[goodk, :] = self.b + np.dot(dEt[:, np.newaxis] * self.Ew[:,
            goodk].T, xbarinv) 
        self.tauh[goodk, :] = self.Ehinvinv[goodk, :]**2 * \
                np.dot(dEtinvinv[:, np.newaxis] * self.Ewinvinv[:, goodk].T,
                        xxtwidinvsq)
        self.tauh[self.tauh < 1e-100] = 0
        self.Eh[goodk, :], self.Ehinv[goodk, :] = _gap.compute_gig_expectations(
                self.b, self.rhoh[goodk, :], self.tauh[goodk, :])
        self.Ehinvinv[goodk, :] = 1./self.Ehinv[goodk, :]

    def update_theta(self):
        goodk = self.goodk()
        xxtwidinvsq = (self.X * self._xtwid(goodk)) ** (-2)
        xbarinv = self._xbar(goodk) ** (-1)
        self.rhot[goodk] = self.beta + np.sum(np.dot(self.Ew[:, goodk].T,
            xbarinv) * self.Eh[goodk, :], axis=1)
        self.taut[goodk] = self.Etinvinv[goodk]**2 * \
                np.sum(np.dot(self.Ewinvinv[:, goodk].T, xxtwidinvsq) *
                        self.Ehinvinv[goodk, :], axis=1)
        self.taut[self.taut < 1e-100] = 0
        self.Et[goodk], self.Etinv[goodk] = _gap.compute_gig_expectations(
                self.beta/self.K, self.rhot[goodk], self.taut[goodk])
        self.Etinvinv[goodk] = 1./self.Etinv[goodk]

    def goodk(self, cut_off=None):
        if cut_off is None:
            cut_off = 1e-10 * np.amax(self.X)

        powers = self.Et * np.amax(self.Ew, axis=0) * np.amax(self.Eh, axis=1)
        sorted = np.flipud(np.argsort(powers))
        idx = np.where(powers[sorted] > cut_off * np.amax(powers))[0]
        goodk = sorted[:(idx[-1] + 1)]
        if powers[goodk[-1]] < cut_off:
            goodk = np.delete(goodk, -1)
        return goodk

    def clear_badk(self):
        ''' Set unsued components' posteriors equal to their priors
        '''
        goodk = self.goodk()
        badk = np.setdiff1d(np.arange(self.K), goodk)
        self.rhow[:, badk] = self.a
        self.tauw[:, badk] = 0
        self.rhoh[badk, :] = self.b
        self.tauh[badk, :] = 0
        self.compute_expectations()

    def bound(self):
        score = 0
        goodk = self.goodk()

        xbar = self._xbar(goodk)
        xtwid = self._xtwid(goodk)
        score -= np.sum(self.X / xtwid + np.log(xbar))
        score += _gap.gig_gamma_term(self.Ew, self.Ewinv, self.rhow, self.tauw,
                self.a, self.a)
        score += _gap.gig_gamma_term(self.Eh, self.Ehinv, self.rhoh, self.tauh,
                self.b, self.b)
        score += _gap.gig_gamma_term(self.Et, self.Etinv, self.rhot, self.taut,
                self.beta/self.K, self.beta)
        return score

    def _xbar(self, goodk=None):
        if goodk is None:
            goodk = np.arange(self.K)
        dEt = self.Et[goodk]
        return np.dot(self.Ew[:, goodk], dEt[:, np.newaxis] * self.Eh[goodk, :])

    def _xtwid(self, goodk):
        dEtinvinv = self.Etinvinv[goodk]
        return np.dot(self.Ewinvinv[:, goodk], dEtinvinv[:, np.newaxis] * 
                self.Ehinvinv[goodk, :])





