"""
CREATED: 2013-05-23 10:58:16 by Dawen Liang <daliang@adobe.com>
"""

import sys, time

import numpy as np
import scipy.optimize as optimize
import scipy.special as special

class VPL:
    def __init__(self, W, L=10, smoothness=100, seed=None):
        self.V = np.log(W)
        self.T, self.F = W.shape
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
        self.gamma = np.random.gamma(smoothness, 2./smoothness, size=(self.F,))

        self.old_U_inc = np.inf
        self.old_alpha_inc = np.inf
        self.old_gamma_inc = np.inf

        # variational parameters and expectations
        self._init_variational(smoothness)

    def _init_variational(self, smoothness):
        self.mu = np.random.randn(self.T, self.L)
        self.sigma = 1./np.random.gamma(smoothness, 1./smoothness, size=(self.T, self.L))
        self.EA, self.EA2, self.ElogA = self._comp_expect(self.mu, self.sigma)

        self.old_mu_inc = np.inf 
        self.old_sigma_inc = np.inf

    def _comp_expect(self, mu, sigma):
        return (np.exp(mu + sigma/2), np.exp(2*mu + 2*sigma), mu)
         
    def vb_e(self, verbose=True, disp=0):
        """ Perform one variational E-step to appxorimate the posterior P(A | -)

        Parameters
        ----------
        verbose: bool
            Output log if true.
        disp: int
            Display warning from solver if > 0, mainly from LBFGS.

        """
        print 'Variational E-step...'
        for l in xrange(self.L):
            self.update_theta(l, disp)
            if verbose and not l % 5:
                sys.stdout.write('.')
        if verbose:
            sys.stdout.write('\n')
        pass

    def update_theta(self, l, disp):                
        def f_stub(theta):
            mu, sigma = theta[:self.T], np.exp(theta[-self.T:])
            Ea, Ea2, Eloga = self._comp_expect(mu, sigma) 

            lcoef = Ea * (np.sum(Eres[t,:] * self.U[l,:] * self.gamma) - self.alpha[l])
            qcoef = -1./2 * Ea2 * np.sum(self.gamma * self.U[l,:]**2)
            return (lcoef, qcoef, Eloga)

        def f(theta):
            lcoef, qcoef, Eloga = f_stub(theta)
            const = self.alpha[l] * Eloga + 1./2 * np.log()
            const = self.alpha[l] * phi
            return -(const + lcoef + qcoef)
                
        def df(theta):
            const = self.alpha[l]
            lcoef, qcoef = _f_stub(phi, t)
            return -(const + lcoef + 2*qcoef)

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        theta0 = np.hstack((self.mu[:,l], np.log(self.sigma[:,l])))

        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warning']:
            theta = np.hstack((self.mu[:,l], np.log(self.sigma[:,l])))
            if d['warnflag'] == 2:
                print 'A[:, {}]: {}, f={}'.format(l, d['task'], f(theta))
            else:
                print 'A[:, {}]: {}, f={}'.format(l, d['warnflag'], f(theta))
            app_grad = approx_grad(f, theta)
            for t in xrange(2 * self.T):
                print 'Theta[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta[t], app_grad[t], df(theta)[t], np.abs(app_grad[t] - df(theta)[t]))

        self.mu[:,l], self.sigma[:,l] = theta_hat[:self.T], np.exp(theta_hat[-self.T:])

        assert(np.all(self.sigma[:,l] > 0))
        self.EA[:,l], self.EA2[:,l], self.ElogA[:,l] = self._comp_expect(self.mu[:,l], self.sigma[:,l])

    def vb_m(self, conv_check=1, atol=0.01, verbose=True, disp=0):
        """ Perform one M-step, update the model parameters with A fixed from E-step

        Parameters
        ----------
        conv_check: int
            Check convergence on the first-order difference if 1 or second-order
            difference if 2. 
        atol: float
            Absolute convergence threshold.
        verbose: bool
            Output log if ture.
        disp: int
            Display warning from solver if > 0, mostly from LBFGS.

        """

        print 'Variational M-step...'
        old_U = self.U.copy()
        old_gamma = self.gamma.copy()
        old_alpha = self.alpha.copy()
        for l in xrange(self.L):
            self.update_u(l, disp)
        self.update_gamma()
        self.update_alpha(disp)
        self._objective_m()
        U_diff = np.mean(np.abs(self.U - old_U))
        sigma_diff = np.mean(np.abs(np.sqrt(1./self.gamma) - np.sqrt(1./old_gamma)))
        alpha_diff = np.mean(np.abs(self.alpha - old_alpha))
        if verbose:
            print 'U increment: {:.4f}\tsigma increment: {:.4f}\talpha increment: {:.4f}'.format(U_diff, sigma_diff, alpha_diff)
        if conv_check == 1:
            if U_diff < atol and sigma_diff < atol and alpha_diff < atol:
                return True
        elif conv_check == 2:
            if self.old_U_inc - U_diff < atol and self.old_gamma_inc - sigma_diff < atol and self.old_alpha_inc - alpha_diff < atol:
                return True
            self.old_U_inc = U_diff
            self.old_gamma_inc = sigma_diff
            self.old_alpha_inc = alpha_diff
        else:
            raise ValueError('conv_check can only be 1 or 2')
        return False

    def update_u(self, l, disp):
        def f(u):
            return np.sum(np.outer(self.EA2[:,l], u**2) - 2*np.outer(self.EA[:,l], u) * Eres)
        
        def df(u):
            tmp = self.EA[:,l]  # for broad-casting
            return np.sum(np.outer(self.EA2[:,l], u) - Eres * tmp[np.newaxis].T, axis=0)

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        u0 = self.U[l,:]
        self.U[l,:], _, d = optimize.fmin_l_bfgs_b(f, u0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'U[{}, :]: {}, f={}'.format(l, d['task'], f(self.U[l,:]))
            else:
                print 'U[{}, :]: {}, f={}'.format(l, d['warnflag'], f(self.U[l,:]))

            app_grad = approx_grad(f, self.U[l,:])
            for idx in xrange(self.F):
                print 'U[{}, {:3d}] = {:.2f}\tApproximated: {:.2f}\tGradient: {:.2f}\t|Approximated - True|: {:.3f}'.format(l, idx, self.U[l,idx], app_grad[idx], df(self.U[l,:])[idx], np.abs(app_grad[idx] - df(self.U[l,:])[idx]))


    def update_gamma(self):
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.gamma = 1./np.mean(self.V**2 - 2 * self.V * EV + EV2, axis=0)

    def update_alpha(self, disp):
        def f(eta):
            tmp1 = np.exp(eta) * eta - special.gammaln(np.exp(eta))
            tmp2 = self.ElogA * (np.exp(eta) - 1) - self.EA * np.exp(eta)
            return -(self.T * tmp1.sum() + tmp2.sum())

        def df(eta):
            return -np.exp(eta) * (self.T * (eta + 1 - special.psi(np.exp(eta))) + np.sum(self.ElogA - self.EA, axis=0))
        
        eta0 = np.log(self.alpha)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.alpha = np.exp(eta_hat)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(f(self.alpha), d['task'])
            else:
                print 'f={}, {}'.format(f(self.alpha), d['warnflag'])
            app_grad = approx_grad(f, self.alpha)
            for l in xrange(self.L):
                print 'Alpha[{:3d}] = {:.2f}\tApproximated: {:.2f}\tGradient: {:.2f}\t|Approximated - True|: {:.3f}'.format(l, self.alpha[l], app_grad[l], df(self.alpha)[l], np.abs(app_grad[l] - df(self.alpha)[l]))


    def _objective_e(self):
        pass

    def _objective_m(self):
        self.obj = 1./2 * self.T * np.sum(np.log(self.gamma))
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.obj -= 1./2 * np.sum((self.V**2 - 2 * self.V * EV + EV2) * self.gamma)
        self.obj += self.T * np.sum(self.alpha * np.log(self.alpha) - special.gammaln(self.alpha))
        self.obj += np.sum(self.ElogA * (self.alpha - 1) - self.EA * self.alpha)


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2*delta)
    return grad
