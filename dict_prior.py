'''
CREATED: 2013-05-23 10:58:16 by Dawen Liang <daliang@adobe.com>
'''

import sys

import numpy as np
import scipy.optimize as optimize
import scipy.special as special

class SF_Dict:
    def __init__(self, W, L=10, smoothness=100, seed=None, **kwargs):
        self.V = np.log(W)
        self.N, self.F = W.shape
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
        self.U = np.random.randn(self.L, self.F)
        self.alpha = np.random.gamma(smoothness, 1./smoothness, size=(self.L,))
        self.gamma = np.random.gamma(smoothness, 1./smoothness, size=(self.F,))

        self.obj = -np.inf

        # variational parameters and expectations
        self._init_variational(smoothness)

    def _init_variational(self, smoothness):
        # just in case seed was previously fixed
        np.random.seed()
        self.mu = np.random.randn(self.N, self.L)
        self.r = np.random.gamma(smoothness, 1./smoothness, size=(self.N, self.L))
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
        if verbose:
            print 'Variational E-step...'
        if e_converge:
            # do e-step until variational inference converges
            self._init_variational(smoothness)
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
                if verbose and not l % 5:
                    sys.stdout.write('.')
            if verbose:
                sys.stdout.write('\n')
        return True

    def update_phi(self, l):                
        def f_stub(phi):
            lcoef = np.sum(np.outer(np.exp(phi), self.U[l,:]) * Eres * self.gamma, axis=1)
            qcoef = -1./2 * np.sum(np.outer(np.exp(2*phi), self.U[l,:]**2) * self.gamma, axis=1)
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

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        phi0 = self.mu[:,l]
        mu_hat, _, d = optimize.fmin_l_bfgs_b(f, phi0, fprime=df, disp=0)

        self.mu[:,l], self.r[:,l] = mu_hat, df2(mu_hat)
        if np.any(self.r[:,l] <= 0):
            if d['warnflag'] == 2:
                print 'A[:, {}]: {}, f={}'.format(l, d['task'], f(mu_hat))
            else:
                print 'A[:, {}]: {}, f={}'.format(l, d['warnflag'], f(mu_hat))
            return False 

        self.EA[:,l], self.EA2[:,l] = self._comp_expect(self.mu[:,l], self.r[:,l])
        return True

    def vb_m(self, verbose=True):
        if verbose:
            print 'Updating U...'
        for l in xrange(self.L):
            self.update_u(l)
        if verbose:
            print 'Updating gamma and alpha...'
        self.update_gamma()
        self.update_alpha()
        pass

    def update_u(self, l):
        def f(u):
            pass
        def df(u):
            pass
        pass

    def update_gamma(self):
        #def f(gamma):
        #    tmp = np.sum(self.V**2 - 2 * self.V * EV + EV2, axis=0) * gamma
        #    return -(self.N * np.sum(np.log(gamma)) - np.sum(tmp))
        #def df(gamma):
        #    return -(self.N / gamma - np.sum(self.V**2 - 2 * self.V * EV + EV2, axis=0))
        #        
        #EV = np.dot(self.EA, self.U)
        #EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        #gamma0 = self.gamma
        #self.gamma, _, d = optimize.fmin_l_bfgs_b(f, gamma0, fprime=df, disp=0)
        #if d['warnflag']:
        #    print 'Warning: gamma is not optimal'
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.gamma = 1./np.mean(self.V**2 - 2 * self.V * EV + EV2, axis=0)

    def update_alpha(self):
        def f(alpha):
            tmp1 = alpha * np.log(alpha) - special.gammaln(alpha)
            tmp2 = self.mu * (alpha - 1) - self.EA * alpha
            return -(self.N * tmp1.sum() + tmp2.sum())
        def df(alpha):
            return -(self.N * (np.log(alpha) + 1 - special.psi(alpha)) + np.sum(self.mu - self.EA, axis=0))

        alpha0 = self.alpha        
        self.alpha, _, d = optimize.fmin_l_bfgs_b(f, alpha0, fprime=df, disp=0)
        if d['warnflag']:
            print 'Warning: alpha is not optimal'

    def _objective(self):
        pass
