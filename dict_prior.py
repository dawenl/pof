'''
CREATED: 2013-05-23 10:58:16 by Dawen Liang <daliang@adobe.com>
'''

import sys

import numpy as np
import scipy.optimize as optimize

class SF_Dict:
    def __init__(self, W, L=10, smoothness=100, seed=None, **kwargs):
        self.V = np.log(W)
        self.F, self.N = W.shape
        self.L = L
        if seed is None:
            print 'Using random seed'
            seed = np.random.seed()
        self._parse_args(**kwargs)
        self._init(smoothness=smoothness)

    def _parse_args(self, **kwargs):
        pass

    def _init(self, smoothness=100):
        # model parameters
        self.U = np.random.randn(self.F, self.L)
        self.alpha = np.random.gamma(smoothness, 1./smoothness, size=(self.L,))
        self.gamma = np.random.gamma(smoothness, 1./smoothness, size=(self.F,))

        # variational parameters and expectations
        self._init_variational(smoothness)

    def _init_variational(self, smoothness):
        # just in case seed was fixed previously
        np.random.seed()
        self.mu = np.random.randn(self.L, self.N)
        self.r = np.random.gamma(smoothness, 1./smoothness, size=(self.L,
            self.N))
        self.EA, self.EA2 = self._comp_expect(self.mu, self.r)
        self.old_mu = np.inf
        self.old_r = np.inf

    def _comp_expect(self, mu, r):
        return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r))
         
    #def update(self, e_converge=True, smoothness=100, verbose=True):
    #    self.vb_e(e_converge, smoothness, verbose)
    #    self.vb_m()
    #    self._objective()

    def vb_e(self, e_converge=True, smoothness=100, verbose=True):
        if e_converge:
            # do e-step until variational inference converges
            self._init_variational(smoothness)
            if verbose:
                print 'Variational E-step...'
            while True:
                for l in xrange(self.L):
                    if not self.update_phi(l):
                        return False
                    if verbose and not l % 5:
                        sys.stdout.write('.')
                if verbose:
                    sys.stdout.write('\n')
                    print 'mu increment: {}; r increment: {}'.format(np.mean(np.abs(self.old_mu - self.mu)), np.mean(np.abs(self.old_r - self.r)))
                if np.mean(np.abs(self.old_mu - self.mu)) <= 1e-3 and np.mean(np.abs(self.old_r - self.r)) <= 1e-3:
                    break
                self.old_mu = self.mu.copy()
                self.old_r = self.r.copy()
        else:
            # do e-step for one iteration
            for l in xrange(self.L):
                if not self.update_phi(l):
                    return False
        return True

    def update_phi(self, l):                
        def f_stub(phi):
            lcoef = np.sum(self.gamma.reshape(-1, 1) * np.outer(self.U[:,l], np.exp(phi)) * Eres, axis=0) 
            qcoef = -1./2 * np.sum(self.gamma.reshape(-1, 1) * np.outer(self.U[:,l]**2, np.exp(2*phi)), axis=0) 
            return (lcoef, qcoef)

        def f(phi):
            const = self.alpha[l] * (phi - np.exp(phi))   
            lcoef, qcoef = f_stub(phi)
            return -np.sum(const + lcoef + qcoef)

        def df(phi):
            const = self.alpha[l] * (1 - np.exp(phi))
            lcoef, qcoef = f_stub(phi)
            return -(const + lcoef + 2*qcoef)
            
        def df2(phi):
            const = -self.alpha[l] * np.exp(phi)
            lcoef, qcoef = f_stub(phi)
            return -(const + lcoef + 4*qcoef)

        Eres = self.V - np.dot(self.U, self.EA) + np.outer(self.U[:,l],
                self.EA[l,:])
        phi0 = self.mu[l,:]
        mu_hat, _, d = optimize.fmin_l_bfgs_b(f, phi0, fprime=df, disp=0)

        self.mu[l,:], self.r[l,:] = mu_hat, df2(mu_hat)
        if np.any(self.r[l,:] <= 0):
            if d['warnflag'] == 2:
                print 'A[{}, :]: {}, f={}'.format(l, d['task'], f(mu_hat))
                print f_stub(mu_hat)
            else:
                print 'A[{}, :]: {}, f={}'.format(l, d['warnflag'], f(mu_hat))
            return False 

        self.EA[l,:], self.EA2[l,:] = self._comp_expect(self.mu[l,:], self.r[l,:])
        return True

    def vb_m(self, verbose):
        if verbose:
            print 'Updating U...'
        for l in xrange(self.L):
            self.update_u(l)
        if verbose:
            print 'Updating gamma and alpha'
        self.update_gamma()
        self.update_alpha()
        pass

    def update_u(self, l):
        pass

    def update_gamma(self):
        pass

    def update_alpha(self):
        pass

    def _objective(self):
        pass
