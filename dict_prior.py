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
        self.gamma = np.random.gamma(smoothness, 1./(2*smoothness), size=(self.F,))

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
         
    def vb_e(self, e_converge=True, smoothness=100, maxiter=500, verbose=True):
        if verbose:
            print 'Variational E-step...'
        if e_converge:
            # do e-step until variational inference converges
            self._init_variational(smoothness)
            for _ in xrange(maxiter):
                for l in xrange(self.L):
                    if not self.update_phi(l):
                        return False
                    if verbose and not l % 5:
                        sys.stdout.write('.')
                if verbose:
                    sys.stdout.write('\n')
                    print 'mu increment: {}; r increment: {}'.format(np.mean(np.abs(self.old_mu - self.mu)), np.mean(np.abs(self.old_r - self.r)))
                if np.mean(np.abs(self.old_mu - self.mu)) <= 1e-3 and np.mean(np.abs(self.old_r - self.r)) <= 5 * 1e-3:
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

    def update_phi(self, l, full_output=False):                
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
                
            app_grad = approx_grad(f, mu_hat)
            for n in xrange(self.N):
                print '|Approximated - True grad[{}]|: {}'.format(n,
                        np.abs(app_grad[n] - df(mu_hat)[n]))
            return False 

        self.EA[:,l], self.EA2[:,l] = self._comp_expect(self.mu[:,l], self.r[:,l])
        return True

    def vb_m(self, verbose=True):
        if verbose:
            print 'Variational M-step...'
        for l in xrange(self.L):
            self.update_u(l)
        self.update_gamma()
        self.update_alpha()
        self._objective()

    def update_u(self, l):
        def f(u):
            return np.sum(np.outer(self.EA2[:,l], u**2) - 2*np.outer(self.EA[:,l], u) * Eres)
        
        def df(u):
            tmp = self.EA[:,l]  # for broad-casting
            return np.sum(np.outer(self.EA2[:,l], u) - Eres * tmp[np.newaxis].T, axis=0)

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        u0 = self.U[l,:]
        self.U[l,:], _, d = optimize.fmin_l_bfgs_b(f, u0, fprime=df, disp=0)
        if d['warnflag']:
            if d['warnflag'] == 2:
                print 'U[{}, :]: {}, f={}'.format(l, d['task'], f(self.U[l,:]))
            else:
                print 'U[{}, :]: {}, f={}'.format(l, d['warnflag'], f(self.U[l,:]))

    def update_gamma(self):
        # closed form update is available for gamma
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
            print 'Warning: alpha is not optimal (Warning type:{})'.format(d['warnflag'])

            app_grad = approx_grad(f, self.alpha)
            for l in xrange(self.L):
                print '|Approximated - True grad[{}]|: {}'.format(l,
                        np.abs(app_grad[l] - df(self.alpha)[l]))

    def _objective(self):
        self.obj = 1./2 * self.N * np.sum(np.log(self.gamma))
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.obj -= np.sum((self.V**2 - 2 * self.V * EV + EV2) * self.gamma)
        self.obj += self.N * np.sum(self.alpha * np.log(self.alpha) -
                special.gammaln(self.alpha))
        self.obj += np.sum(self.mu * (self.alpha - 1) - self.EA * self.alpha)


def approx_grad(f, x, delta=1e-6):
    grad = np.zeros_like(x)
    for i, _ in enumerate(x):
        tmpx = x.copy()
        tmpx[i] += delta
        grad[i] = (f(tmpx) - f(x)) / delta
    return grad
