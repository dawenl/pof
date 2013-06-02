'''
CREATED: 2013-05-23 10:58:16 by Dawen Liang <daliang@adobe.com>
'''

import sys, time

import numpy as np
import scipy.optimize as optimize
import scipy.special as special

class SF_Dict:
    def __init__(self, W, L=10, smoothness=100, seed=None):
        self.V = np.log(W)
        self.N, self.F = W.shape
        self.L = L
        if seed is None:
            print 'Using random seed'
            np.random.seed()
        else:
            print 'Using fixed seed {}'.format(seed)
            np.random.seed(seed) 
        self._init(smoothness=smoothness)

    def _init(self, smoothness=100):
        # model parameters
        self.U = np.random.randn(self.L, self.F)
        self.alpha = np.random.gamma(smoothness, 1./smoothness, size=(self.L,))
        self.gamma = np.random.gamma(smoothness, 1./(2*smoothness), size=(self.F,))

        # variational parameters and expectations
        self._init_variational(smoothness)

    def _init_variational(self, smoothness):
        self.mu = np.random.randn(self.N, self.L)
        self.r = np.random.gamma(smoothness, 1./smoothness, size=(self.N, self.L))
        self.EA, self.EA2, self.ElogA = self._comp_expect(self.mu, self.r)

    def _comp_expect(self, mu, r):
        return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r), mu)
         
    def vb_e(self, e_converge=True, smoothness=100, maxiter=500, atol=1e-3, verbose=True):
        print 'Variational E-step...'
        if e_converge:
            # do e-step until variational inference converges
            self._init_variational(smoothness)
            for _ in xrange(maxiter):
                old_mu = self.mu.copy()
                old_r = self.r.copy()
                start_t = time.time()
                for l in xrange(self.L):
                    self.update_phi(l)
                    if verbose and not l % 5:
                        sys.stdout.write('.')
                t = time.time() - start_t
                mu_diff = np.mean(np.abs(old_mu - self.mu))
                sigma_diff = np.mean(np.abs(np.sqrt(1./old_r) - np.sqrt(1./self.r)))
                if verbose:
                    sys.stdout.write('\n')
                    print 'mu increment: {:.4f}\tsigma increment: {:.4f}\ttime: {:.2f}'.format(mu_diff, sigma_diff, t)
                if mu_diff <= atol and sigma_diff <= atol:
                    break
        else:
            # do e-step for one iteration
            for l in xrange(self.L):
                self.update_phi(l)
                if verbose and not l % 5:
                    sys.stdout.write('.')
            if verbose:
                sys.stdout.write('\n')

    def update_phi(self, l):                
        def _f_stub(phi, n):
            lcoef = np.exp(phi) * (np.sum(Eres[n,:] * self.U[l,:] * self.gamma) - self.alpha[l])
            qcoef = -1./2 * np.exp(2*phi) * np.sum(self.gamma * self.U[l,:]**2)
            return (lcoef, qcoef)

        def _f(phi, n):
            const = self.alpha[l] * phi
            lcoef, qcoef = _f_stub(phi, n)
            return -(const + lcoef + qcoef)
                
        def _df(phi, n):
            const = self.alpha[l]
            lcoef, qcoef = _f_stub(phi, n)
            return -(const + lcoef + 2*qcoef)

        def _df2(phi, n):
            const = 0
            lcoef, qcoef = _f_stub(phi, n)
            return -(const + lcoef + 4*qcoef)

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        for n in xrange(self.N): 
            self.mu[n, l], _, d = optimize.fmin_l_bfgs_b(_f, self.mu[n, l], fprime=_df, args=(n,), disp=0)
            self.r[n, l] = _df2(self.mu[n, l], n)
            if d['warnflag']:
                if d['warnflag'] == 2:
                    print 'Phi[{}, {}]: {}, f={}'.format(n, l, d['task'], _f(self.mu[n, l], n))
                else:
                    print 'Phi[{}, {}]: {}, f={}'.format(n, l, d['warnflag'], _f(self.mu[n, l], n))
                app_grad = approx_grad(_f, self.mu[n, l], args=(n,))[0]
                app_hessian = approx_grad(_df, self.mu[n, l], args=(n,))[0]
                print '\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(app_grad, _df(self.mu[n, l], n), np.abs(app_grad - _df(self.mu[n, l], n)))
                print '\tApproximated: {:.5f}\tHessian: {:.5f}\t|Approximated - True|: {:.5f}'.format(app_hessian, _df2(self.mu[n, l], n), np.abs(app_hessian - _df2(self.mu[n, l], n)))

        assert(np.all(self.r[:,l] > 0))
        self.EA[:,l], self.EA2[:,l], self.ElogA[:,l] = self._comp_expect(self.mu[:,l], self.r[:,l])

    def vb_m(self, atol=5*1e-3, verbose=True):
        print 'Variational M-step...'
        old_U = self.U.copy()
        old_gamma = self.gamma.copy()
        old_alpha = self.alpha.copy()
        for l in xrange(self.L):
            self.update_u(l)
        self.update_gamma()
        self.update_alpha()
        self._objective()
        U_diff = np.mean(np.abs(self.U - old_U))
        sigma_diff = np.mean(np.abs(np.sqrt(1./self.gamma) - np.sqrt(1./old_gamma)))
        alpha_diff = np.mean(np.abs(self.alpha - old_alpha))
        if verbose:
            print 'U increment: {:.4f}\tsigma increment: {:.4f}\talpha increment: {:.4f}'.format(U_diff, sigma_diff, alpha_diff)
        if U_diff < atol and sigma_diff < atol and alpha_diff < atol:
            return True
        return False

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

            app_grad = approx_grad(f, self.U[l,:])
            for idx in xrange(self.F):
                print 'U[{}, {:3d}] = {:.2f}\tApproximated: {:.2f}\tTrue: {:.2f}\t|Approximated - True|: {:.3f}'.format(l, idx, self.U[l,idx], app_grad[idx], df(self.U[l,:])[idx], np.abs(app_grad[idx] - df(self.U[l,:])[idx]))


    def update_gamma(self):
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.gamma = 1./np.mean(self.V**2 - 2 * self.V * EV + EV2, axis=0)

    def update_alpha(self):
        def f(eta):
            tmp1 = np.exp(eta) * eta - special.gammaln(np.exp(eta))
            tmp2 = self.ElogA * (np.exp(eta) - 1) - self.EA * np.exp(eta)
            return -(self.N * tmp1.sum() + tmp2.sum())

        def df(eta):
            return -np.exp(eta) * (self.N * (eta + 1 - special.psi(np.exp(eta))) + np.sum(self.ElogA - self.EA, axis=0))
        
        eta0 = np.log(self.alpha)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.alpha = np.exp(eta_hat)
        if d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(f(self.alpha), d['task'])
            else:
                print 'f={}, {}'.format(f(self.alpha), d['warnflag'])
            app_grad = approx_grad(f, self.alpha)
            for l in xrange(self.L):
                print 'Alpha[{:3d}] = {:.2f}\tApproximated: {:.2f}\tTrue: {:.2f}\t|Approximated - True|: {:.3f}'.format(l, self.alpha[l], app_grad[l], df(self.alpha)[l], np.abs(app_grad[l] - df(self.alpha)[l]))

    def _objective(self):
        self.obj = 1./2 * self.N * np.sum(np.log(self.gamma))
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.obj -= 1./2 * np.sum((self.V**2 - 2 * self.V * EV + EV2) * self.gamma)
        self.obj += self.N * np.sum(self.alpha * np.log(self.alpha) - special.gammaln(self.alpha))
        self.obj += np.sum(self.ElogA * (self.alpha - 1) - self.EA * self.alpha)


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2*delta)
    return grad
