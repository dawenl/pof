"""
CREATED: 2013-06-14 10:31:54 by Dawen Liang <daliang@adobe.com>   

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

        #self.old_mu_inc = np.inf 
        #self.old_sigma_inc = np.inf

    def _comp_expect(self, mu, sigma):
        return (np.exp(mu + sigma/2), np.exp(2*mu + 2*sigma), mu)
         
    def vb_e(self, cold_start=True, smoothness=100, verbose=True, disp=0):
        """ Perform one variational E-step to appxorimate the posterior P(A | -)

        Parameters
        ----------
        cold_start: bool
            Do e-step with fresh initialization until convergence if true,
            otherwise just do one sub-iteration with previous values as
            initialization.
        smoothness: float
            Smootheness of the variational initialization, larger value will
            lead to more concentrated initialization.
        verbose: bool
            Output log if true.
        disp: int
            Display warning from solver if > 0, mainly from LBFGS.

        """
        print 'Variational E-step...'
        if cold_start:
            self._init_variational(smoothness)
        start_t = time.time()
        for l in xrange(self.L):
            self.update_theta(l, disp)
            if verbose and not l % 5:
                sys.stdout.write('.')
        t = time.time() - start_t
        if verbose:
            #sys.stdout.write('\n')
            print '\ttime: {:.2f}'.format(t)
        pass

    def update_theta(self, l, disp):                
        def f(theta):
            mu, sigma = theta[:self.T], np.exp(theta[-self.T:])
            Ea, Ea2, Eloga = self._comp_expect(mu, sigma) 

            const = (self.alpha[l] - 1) * Eloga + np.log(sigma)/2 + mu
            return -np.sum(const + Ea * lcoef + Ea2 * qcoef)
                
        def df(theta):
            mu, sigma = theta[:self.T], np.exp(theta[-self.T:])
            Ea, Ea2, _ = self._comp_expect(mu, sigma) 

            grad_mu = Ea * lcoef + 2 * Ea2 * qcoef + self.alpha[l]   
            grad_sigma = sigma * (Ea * lcoef/2 + 2 * Ea2 * qcoef + 1./sigma) 
            return -np.hstack((grad_mu, grad_sigma))

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        lcoef = np.sum(Eres * self.U[l, :] * self.gamma, axis=1) - self.alpha[l]
        qcoef = -np.sum(self.gamma * self.U[l,:]**2)/2

        theta0 = np.hstack((self.mu[:,l], np.log(self.sigma[:,l])))

        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'A[:, {}]: {}, f={}'.format(l, d['task'], f(theta_hat))
            else:
                print 'A[:, {}]: {}, f={}'.format(l, d['warnflag'], f(theta_hat))
            app_grad = approx_grad(f, theta_hat)
            for t in xrange(self.T):
                print 'mu[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[t], app_grad[t], df(theta_hat)[t], np.abs(app_grad[t] - df(theta_hat)[t]))
            for t in xrange(self.T):
                print 'sigma[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[t + self.T], app_grad[t + self.T], df(theta_hat)[t + self.T], np.abs(app_grad[t + self.T] - df(theta_hat)[t + self.T]))

        self.mu[:,l], self.sigma[:,l] = theta_hat[:self.T], np.exp(theta_hat[-self.T:])

        #def _f(theta, t):
        #    mu, sigma = theta[0], np.exp(theta[1])
        #    Ea, Ea2, Eloga = self._comp_expect(mu, sigma) 

        #    const = (self.alpha[l] - 1) * Eloga + np.log(sigma)/2 + mu
        #    return -np.sum(const + Ea * lcoef[t] + Ea2 * qcoef)
        #        
        #def _df(theta, t):
        #    mu, sigma = theta[0], np.exp(theta[1])
        #    Ea, Ea2, _ = self._comp_expect(mu, sigma) 

        #    grad_mu = Ea * lcoef[t] + 2 * Ea2 * qcoef + self.alpha[l]   
        #    grad_sigma = sigma * (Ea * lcoef[t]/2 + 2 * Ea2 * qcoef + 1./sigma) - .5
        #    return -np.array([grad_mu, grad_sigma])

        #Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        #lcoef = np.sum(Eres * self.U[l, :] * self.gamma, axis=1) - self.alpha[l]
        #qcoef = -np.sum(self.gamma * self.U[l,:]**2)/2

        #for t in xrange(self.T):
        #    theta0 = np.array([self.mu[t,l], np.log(self.sigma[t,l])])
        #    theta_hat, _, d = optimize.fmin_l_bfgs_b(_f, theta0, fprime=_df, args=(t,), disp=0)
        #    if disp and d['warnflag']:
        #        if d['warnflag'] == 2:
        #            print 'A[:, {}]: {}, f={}'.format(l, d['task'], _f(theta_hat, t))
        #        else:
        #            print 'A[:, {}]: {}, f={}'.format(l, d['warnflag'], _f(theta_hat, t))
        #        app_grad = approx_grad(_f, theta_hat, args=(t,))
        #        print 'mu[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[0], app_grad[0], _df(theta_hat, t)[0], np.abs(app_grad[0] - _df(theta_hat, t)[0]))
        #        print 'sigma[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[1], app_grad[1], _df(theta_hat, t)[1], np.abs(app_grad[1] - _df(theta_hat, t)[1]))

        #    self.mu[t,l], self.sigma[t,l] = theta_hat[0], np.exp(theta_hat[1])

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
