"""
CREATED: 2013-06-24 16:12:52 by Dawen Liang <daliang@adobe.com>

Source-filter dictionary prior learning with gamma variational distribution

"""

import sys
import time

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
        self._init(smoothness)
        self._init_variational(smoothness)

    def _init(self, smoothness):
        # model parameters
        self.U = np.random.randn(self.L, self.F)
        self.alpha = np.random.gamma(smoothness,
                                     1. / smoothness,
                                     size=(self.L,))
        self.gamma = np.random.gamma(smoothness,
                                     2. / smoothness,
                                     size=(self.F,))

    def _init_variational(self, smoothness):
        self.a = np.random.gamma(smoothness,
                                 1. / smoothness,
                                 size=(self.T, self.L))
        b = np.random.gamma(smoothness,
                            1. / smoothness,
                            size=(self.T, self.L))
        self.mu = self.a / b
        self.EA, self.EA2, self.ElogA = comp_expect(self.a, self.mu)

    def vb_e(self, cold_start=True, smoothness=100, verbose=True, disp=0):
        """ Perform one variational E-step, which may have one sub-iteration or
        multiple sub-iterations if e_converge is set to True, to appxorimate
        the posterior P(A | -)

        Parameters
        ----------
        cold_start: bool
            Do e-step with fresh start, otherwise just do e-step with
            previous values as initialization.
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
            # re-initialize all the variational parameters
            self._init_variational(smoothness)

        start_t = time.time()
        for t in xrange(self.T):
            self.update_theta_batch(t, disp)
            if verbose and not t % 100:
                sys.stdout.write('.')
        t = time.time() - start_t
        if verbose:
            sys.stdout.write('\n')
            print 'Batch update\ttime: {:.2f}'.format(t)

    def update_theta_batch(self, t, disp):
        def f(theta):
            a, mu = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Ea2, Eloga = comp_expect(a, mu)

            Ev = np.dot(Ea, self.U)
            Ev2 = np.dot(Ea2, self.U**2) + Ev**2 - np.dot(Ea**2, self.U**2)

            likeli = .5 * (2 * self.V[t] * Ev - Ev2) * self.gamma
            prior = (self.alpha - 1) * Eloga - self.alpha * Ea
            ent = entropy(a, mu)

            return -(likeli.sum() + prior.sum() + ent.sum())

        def df(theta):
            a, mu = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Ea2, _ = comp_expect(a, mu)

            Eres = self.V[t] - np.dot(Ea, self.U) + self.U * Ea[:, np.newaxis]
            lcoef = np.sum(self.U * Eres * self.gamma, axis=1) - self.alpha
            grad_a = a * (-mu**2/a**2 * qcoef/2 + (self.alpha - a) *
                          special.polygamma(1, a) - self.alpha / a + 1)
            grad_mu = mu * (lcoef + (mu + mu / a) * qcoef + self.alpha / mu)

            return -np.hstack((grad_a, grad_mu))

        qcoef = -np.sum(self.U**2 * self.gamma, axis=1)

        theta0 = np.hstack((np.log(self.a[t]), np.log(self.mu[t])))
        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'A[{}, :]: {}, f={}'.format(t, d['task'],
                                                  f(theta_hat))
            else:
                print 'A[{}, :]: {}, f={}'.format(t, d['warnflag'],
                                                  f(theta_hat))
            app_grad = approx_grad(f, theta_hat)
            ana_grad = df(theta_hat)
            for l in xrange(self.L):
                print_gradient('log_a[{}, {:3d}]'.format(t, l),
                               theta_hat[l], ana_grad[l], app_grad[l])
                print_gradient('log_mu[{}, {:3d}]'.format(t, l),
                               theta_hat[l + self.L], ana_grad[l + self.L],
                               app_grad[l + self.L])

        self.a[t], self.mu[t] = np.exp(theta_hat[:self.L]), np.exp(
            theta_hat[-self.L:])
        assert(np.all(self.a[t] > 0))
        assert(np.all(self.mu[t] > 0))
        self.EA[t], self.EA2[t], self.ElogA[t] = comp_expect(
            self.a[t], self.mu[t])

    def vb_m(self, batch=False, verbose=True, disp=0, update_alpha=True):
        """ Perform one M-step, update the model parameters with A fixed from
        E-step

        Parameters
        ----------
        batch: bool
            Update U as a whole optimization if true. Otherwise, update U
            across different basis.
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
            for f in xrange(self.F):
                self.update_u_batch(f, disp)
        else:
            for l in xrange(self.L):
                self.update_u(l, disp)
        self.update_gamma()
        if update_alpha:
            self.update_alpha(disp)
        U_diff = np.mean(np.abs(self.U - old_U))
        sigma_diff = np.mean(np.abs(np.sqrt(1. / self.gamma) -
                                    np.sqrt(1. / old_gamma)))
        alpha_diff = np.mean(np.abs(self.alpha - old_alpha))
        if verbose:
            print('U increment: {:.4f}\tsigma increment: {:.4f}\t'
                  'alpha increment: {:.4f}'.format(U_diff,
                                                   sigma_diff,
                                                   alpha_diff))

    def update_u_batch(self, f, disp):
        def fun(u):
            Ev = np.dot(self.EA, u)
            Ev2 = np.dot(self.EA2, u**2) + Ev**2 - np.dot(self.EA**2, u**2)
            return np.sum(Ev2 - 2 * self.V[:, f] * Ev)

        def dfun(u):
            Eres = self.V[:, f, np.newaxis] - np.dot(self.EA, u)[:, np.newaxis]
            Eres = Eres + self.EA * u
            return np.sum(self.EA2 * u - self.EA * Eres, axis=0)

        u0 = self.U[:, f]
        self.U[:, f], _, d = optimize.fmin_l_bfgs_b(fun, u0, fprime=dfun,
                                                    disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'U[:, {}]: {}, f={}'.format(f, d['task'],
                                                  fun(self.U[:, f]))
            else:
                print 'U[:, {}]: {}, f={}'.format(f, d['warnflag'],
                                                  fun(self.U[:, f]))
            app_grad = approx_grad(fun, self.U[:, f])
            ana_grad = dfun(self.U[:, f])
            for l in xrange(self.L):
                print_gradient('U[{}, {:3d}]'.format(l, f), self.U[l, f],
                               ana_grad[l], app_grad[l])

    def update_u(self, l, disp):
        def f(u):
            return np.sum(np.outer(self.EA2[:, l], u**2) -
                          2 * np.outer(self.EA[:, l], u) * Eres)

        def df(u):
            return np.sum(np.outer(self.EA2[:, l], u) -
                          Eres * self.EA[:, l, np.newaxis], axis=0)

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:, l],
                                                           self.U[l])
        u0 = self.U[l]
        self.U[l], _, d = optimize.fmin_l_bfgs_b(f, u0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'U[{}, :]: {}, f={}'.format(l, d['task'],
                                                  f(self.U[l]))
            else:
                print 'U[{}, :]: {}, f={}'.format(l, d['warnflag'],
                                                  f(self.U[l]))

            app_grad = approx_grad(f, self.U[l])
            ana_grad = df(self.U[l])
            for idx in xrange(self.F):
                print_gradient('U[{}, {:3d}]'.format(l, idx), self.U[l, idx],
                               ana_grad[idx], app_grad[idx])

    def update_gamma(self):
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2,
                                                           self.U**2)
        self.gamma = 1. / np.mean(self.V**2 - 2 * self.V * EV + EV2, axis=0)

    def update_alpha(self, disp):
        def f(eta):
            tmp1 = np.exp(eta) * eta - special.gammaln(np.exp(eta))
            tmp2 = self.ElogA * (np.exp(eta) - 1) - self.EA * np.exp(eta)
            return -(self.T * tmp1.sum() + tmp2.sum())

        def df(eta):
            return -np.exp(eta) * (self.T * (eta + 1 -
                                             special.psi(np.exp(eta)))
                                   + np.sum(self.ElogA - self.EA, axis=0))

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
                print_gradient('Alpha[{:3d}]'.format(l), self.alpha[l],
                               df(self.alpha)[l], app_grad[l])

    def bound(self):
        bound = .5 * self.T * np.sum(np.log(self.gamma))
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2,
                                                           self.U**2)
        bound -= .5 * np.sum((self.V**2 - 2 * self.V * EV + EV2) * self.gamma)
        bound += self.T * np.sum(self.alpha * np.log(self.alpha) -
                                 special.gammaln(self.alpha))
        bound += np.sum(self.ElogA * (self.alpha - 1) - self.EA * self.alpha)
        bound += np.sum(entropy(self.a, self.mu))
        return bound


def print_gradient(name, val, grad, approx):
    print('{} = {:.2f}\tGradient: {:.2f}\tApprox: {:.2f}\t'
          '| Diff |: {:.3f}'.format(name, val, grad, approx,
                                    np.abs(grad - approx)))


def comp_expect(alpha, mu):
    return (mu, mu**2 + mu**2 / alpha, special.psi(alpha) - np.log(alpha / mu))


def entropy(alpha, mu):
    return (alpha - np.log(alpha / mu) + special.gammaln(alpha) +
            (1 - alpha) * special.psi(alpha))


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2 * delta)
    return grad
