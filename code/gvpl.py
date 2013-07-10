"""
CREATED: 2013-06-24 16:12:52 by Dawen Liang <daliang@adobe.com> 

Source-filter dictionary prior learning with gamma variational distribution

"""

import sys, time

import numpy as np
import scipy.optimize as optimize
import scipy.special as special

class SF_Dict(object):
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

        # variational parameters and expectations
        self._init_variational(smoothness)

    def _init_variational(self, smoothness):
        self.a = np.random.gamma(smoothness, 1./smoothness, size=(self.T, self.L))
        b = np.random.gamma(smoothness, 1./smoothness, size=(self.T, self.L))
        self.mu = self.a / b 
        self.EA, self.EA2, self.ElogA = comp_expect(self.a, self.mu)

    def vb_e(self, cold_start=True, batch=True, smoothness=100, maxiter=500,
            atol=1e-3, rtol=1e-5, verbose=True, disp=0):
        """ Perform one variational E-step, which may have one sub-iteration or
        multiple sub-iterations if e_converge is set to True, to appxorimate the 
        posterior P(A | -)

        Parameters
        ----------
        cold_start: bool
            Do e-step with fresh start, otherwise just do e-step with 
            previous values as initialization.
        batch: bool
            Do e-step as a whole optimization if true. Otherwise, do multiple
            sub-iterations until convergence.
        smoothness: float
            Smootheness of the variational initialization, larger value will
            lead to more concentrated initialization.
        maxiter: int
            Maximal number of sub-iterations in one e-step.
        atol: float 
            Absolute convergence threshold. 
        rtol: float
            Relative increase convergence threshold.
        verbose: bool
            Output log if true.
        disp: int
            Display warning from solver if > 0, mainly from LBFGS.

        """
        print 'Variational E-step...'
        if cold_start:
            # re-initialize all the variational parameters
            self._init_variational(smoothness)

        if batch:
            start_t = time.time()
            for t in xrange(self.T):
                self.update_theta_batch(t, disp)
                if verbose and not t % 100:
                    sys.stdout.write('.')
            t = time.time() - start_t
            if verbose:
                sys.stdout.write('\n')
                print 'Batch update\ttime: {:.2f}'.format(t)
        else:
            old_bound = -np.inf
            for i in xrange(maxiter):
                old_a = self.a.copy()
                old_mu = self.mu.copy()
                start_t = time.time()
                for l in xrange(self.L):
                    self.update_theta(l, disp)
                    if verbose and not l % 5:
                        sys.stdout.write('.')
                t = time.time() - start_t
                a_diff = np.mean(np.abs(old_a - self.a))
                mu_diff = np.mean(np.abs(old_mu - self.mu))

                self._vb_bound()
                improvement = (self.bound - old_bound) / np.abs(self.bound)

                if verbose:
                    sys.stdout.write('\n')
                    print 'Subiter: {:3d}\ta diff: {:.4f}\tmu diff: {:.4f}\tbound: {:.2f}\tbound improvement: {:.5f}\ttime: {:.2f}'.format(i, a_diff, mu_diff, self.bound, improvement, t)

                if improvement < rtol or (a_diff <= atol and mu_diff <= atol):
                    break
                old_bound = self.bound

    def update_theta_batch(self, t, disp):
        def f(theta):
            a, mu = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Ea2, Eloga = comp_expect(a, mu)

            Ev = np.dot(Ea, self.U)
            Ev2 = np.dot(Ea2, self.U**2) + Ev**2 - np.dot(Ea**2, self.U**2)

            likeli = (2*self.V[t,:]*Ev - Ev2) * self.gamma/2
            prior = (self.alpha - 1) * Eloga - self.alpha * Ea
            ent = entropy(a, mu) 

            return -(likeli.sum() + prior.sum() + ent.sum())

        def df(theta):
            a, mu = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Ea2, _ = comp_expect(a, mu)

            Eres = self.V[t,:] - np.dot(Ea, self.U) + self.U * Ea[:,np.newaxis]
            lcoef = np.sum(self.U * Eres * self.gamma, axis=1) - self.alpha 
            grad_a = a * (-mu**2/a**2 * qcoef/2 + (self.alpha - a) * special.polygamma(1, a) - self.alpha / a + 1)
            grad_mu = mu * (lcoef + (mu + mu/a) * qcoef + self.alpha/mu) 

            return -np.hstack((grad_a, grad_mu))

        qcoef = -np.sum(self.U**2 * self.gamma, axis=1)

        theta0 = np.hstack((np.log(self.a[t,:]), np.log(self.mu[t,:])))
        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'A[{}, :]: {}, f={}'.format(t, d['task'], f(theta_hat))
            else:
                print 'A[{}, :]: {}, f={}'.format(t, d['warnflag'], f(theta_hat))
            app_grad = approx_grad(f, theta_hat)
            for l in xrange(self.L):
                print 'a[{}, {:3d}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[l], app_grad[l], df(theta_hat)[l], np.abs(app_grad[l] - df(theta_hat)[l]))
                print 'mu[{}, {:3d}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[l + self.L], app_grad[l + self.L], df(theta_hat)[l + self.L], np.abs(app_grad[l + self.L] - df(theta_hat)[l + self.L]))

        self.a[t,:], self.mu[t,:] = np.exp(theta_hat[:self.L]), np.exp(theta_hat[-self.L:])
        assert(np.all(self.a[t,:] > 0))
        assert(np.all(self.mu[t,:] > 0))
        self.EA[t,:], self.EA2[t,:], self.ElogA[t,:] = comp_expect(self.a[t,:], self.mu[t,:])

    def update_theta(self, l, disp):                
        def f(theta):
            a, mu = np.exp(theta[:self.T]), np.exp(theta[-self.T:])
            Ea, Ea2, Eloga = comp_expect(a, mu)

            const = (self.alpha[l] - 1) * Eloga + entropy(a, mu) 
            return -np.sum(const + Ea * lcoef + Ea2 * qcoef)
                
        def df(theta):
            a, mu = np.exp(theta[:self.T]), np.exp(theta[-self.T:])

            grad_a = a * (-mu**2/a**2 * qcoef + (self.alpha[l] - a) * special.polygamma(1, a) - self.alpha[l] / a + 1)
            grad_mu = mu * (lcoef + 2 * (mu + mu/a) * qcoef + self.alpha[l]/mu) 
            return -np.hstack((grad_a, grad_mu))

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        lcoef = np.sum(Eres * self.U[l, :] * self.gamma, axis=1) - self.alpha[l]
        qcoef = -np.sum(self.gamma * self.U[l,:]**2)/2

        theta0 = np.hstack((np.log(self.a[:,l]), np.log(self.mu[:,l])))

        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'A[:, {}]: {}, f={}'.format(l, d['task'], f(theta_hat))
            else:
                print 'A[:, {}]: {}, f={}'.format(l, d['warnflag'], f(theta_hat))
            app_grad = approx_grad(f, theta_hat)
            for t in xrange(self.T):
                print 'a[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[t], app_grad[t], df(theta_hat)[t], np.abs(app_grad[t] - df(theta_hat)[t]))
            for t in xrange(self.T):
                print 'mu[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[t + self.T], app_grad[t + self.T], df(theta_hat)[t + self.T], np.abs(app_grad[t + self.T] - df(theta_hat)[t + self.T]))

        self.a[:,l], self.mu[:,l] = np.exp(theta_hat[:self.T]), np.exp(theta_hat[-self.T:])

        assert(np.all(self.a[:,l] > 0))
        assert(np.all(self.mu[:,l] > 0))
        self.EA[:,l], self.EA2[:,l], self.ElogA[:,l] = comp_expect(self.a[:,l], self.mu[:,l])

    def vb_m(self, batch=False, atol=1e-3, verbose=True, disp=0, update_alpha=True):
        """ Perform one M-step, update the model parameters with A fixed from E-step

        Parameters
        ----------
        batch: bool
            Update U as a whole optimization if true. Otherwise, update U across
            different basis.
        atol: float
            Absolute convergence threshold.
        verbose: bool
            Output log if ture.
        disp: int
            Display warning from solver if > 0, mostly from LBFGS.
        update_alpha: bool
            Update alpha if true.

        """

        print 'Variational M-step...'
        old_U = self.U.copy()
        old_gamma = self.gamma.copy()
        old_alpha = self.alpha.copy()
        if batch:
            self.update_u_batch(disp) 
        else:
            for l in xrange(self.L):
                self.update_u(l, disp)
        self.update_gamma()
        if update_alpha:
            self.update_alpha(disp)
        self._objective()
        U_diff = np.mean(np.abs(self.U - old_U))
        sigma_diff = np.mean(np.abs(np.sqrt(1./self.gamma) - np.sqrt(1./old_gamma)))
        alpha_diff = np.mean(np.abs(self.alpha - old_alpha))
        if verbose:
            print 'U increment: {:.4f}\tsigma increment: {:.4f}\talpha increment: {:.4f}'.format(U_diff, sigma_diff, alpha_diff)
        if U_diff < atol and sigma_diff < atol and alpha_diff < atol:
            return True
        return False

    def update_u_batch(self, disp):
        def f(u):
            U = u.reshape(self.L, self.F) 
            EV = np.dot(self.EA, U)
            EV2 = np.dot(self.EA2, U**2) + EV**2 - np.dot(self.EA**2, U**2)
            return -np.sum(2*self.V * EV - EV2)

        def df(u):
            U = u.reshape(self.L, self.F)
            grad_U = np.zeros_like(U)
            for l in xrange(self.L):
                Eres = self.V - np.dot(self.EA, U) + np.outer(self.EA[:,l], U[l,:])
                grad_U[l,:] = np.sum(np.outer(self.EA2[:,l], U[l,:]) - Eres * self.EA[:,l][np.newaxis].T, axis=0)
            return grad_U.ravel()

        u0 = self.U.ravel()
        u_hat, _, d = optimize.fmin_l_bfgs_b(f, u0, fprime=df, disp=0)
        self.U = u_hat.reshape(self.L, self.F)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'U: {}, f={}'.format(d['task'], f(u_hat))
            else:
                print 'U: {}, f={}'.format(d['warnflag'], f(u_hat))

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


    def _vb_bound(self):
        self.bound = np.sum(entropy(self.a, self.mu)) 
        self.bound += np.sum(self.ElogA * (self.alpha - 1) - self.EA * self.alpha)
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.bound += 1./2 * np.sum((2 * EV * self.V - EV2) * self.gamma)

    def _objective(self):
        self.obj = 1./2 * self.T * np.sum(np.log(self.gamma))
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.obj -= 1./2 * np.sum((self.V**2 - 2 * self.V * EV + EV2) * self.gamma)
        self.obj += self.T * np.sum(self.alpha * np.log(self.alpha) - special.gammaln(self.alpha))
        self.obj += np.sum(self.ElogA * (self.alpha - 1) - self.EA * self.alpha)


def comp_expect(alpha, mu):
    return (mu, mu**2 + mu**2/alpha, special.psi(alpha) - np.log(alpha/mu))


def entropy(alpha, mu):
    return (alpha - np.log(alpha / mu) + special.gammaln(alpha) + (1-alpha) * special.psi(alpha))


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2*delta)
    return grad

