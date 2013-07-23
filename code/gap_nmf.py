import numpy as np

import _gap

class GaP_NMF:
    def __init__(self, X, K=100, smoothness=100, seed=None, **kwargs):
        self.X = X / np.mean(X)
        self.K = K 
        F, T = X.shape
        if seed is None:
            print 'Using random seed.'
            np.random.seed()
        else:
            print 'Using fixed seed {}'.format(seed)
            np.random.seed(seed)
        self._parse_args(**kwargs)
        self._init(smoothness)

    def _parse_args(self, **kwargs):
        self.a = kwargs['a'] if 'a' in kwargs else 0.1 
        self.b = kwargs['b'] if 'b' in kwargs else 0.1
        self.alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.

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
        self.compute_expectations()
                
    def compute_expectations(self):
        self.Ew, self.Ewinv = _gap.compute_gig_expectations(self.a, self.rhow,
                self.tauw)
        self.Ewinvinv = 1./self.Ewinv
        self.Eh, self.Ehinv = _gap.compute_gig_expectations(self.b, self.rhoh,
                self.tauh)
        self.Ehinvinv = 1./self.Ehinv
        self.Et, self.Etinv = _gap.compute_gig_expectations(self.alpha/self.K,
                self.rhot, self.taut)
        self.Etinvinv = 1./self.Etinv

    def update(self):
        self.update_h()
        self.update_w()
        self.update_theta()

        self.clear_badk()
        pass

    def update_w(self):
        pass

    def update_h(self):
        pass

    def update_theta(self):
        pass

    def good_k(self):
        pass

    def clear_badk(self):
        pass

    def bound(self):
        pass

    def _xtwid(self, goodk):
        return np.dot(self.Ewinvinv[:, goodk], self.Etinvinv[goodk] * 
                self.Ehinvinv[goodk, :])




