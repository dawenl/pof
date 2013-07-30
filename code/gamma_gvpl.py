"""

Source-filter dictionary prior learning for gamma noise model

CREATED: 2013-07-12 11:09:44 by Dawen Liang <daliang@adobe.com>

"""

import sys
import time

import numpy as np
import scipy.optimize as optimize
import scipy.special as special


class SF_Dict(object):
    def __init__(self, W, L=10, smoothness=100, seed=None):
        self.W = W.copy()
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
                                     1. / smoothness,
                                     size=(self.F,))

    def _init_variational(self, smoothness):
        self.a = smoothness * np.random.gamma(smoothness,
                                              1. / smoothness,
                                              size=(self.T, self.L))
        self.b = smoothness * np.random.gamma(smoothness,
                                              1. / smoothness,
                                              size=(self.T, self.L))
        self.EA, self.ElogA = comp_expect(self.a, self.b)

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

        if verbose:
            last_score = self.bound()
            print('Update (initial)\tObj: {:.2f}'.format(last_score))
            start_t = time.time()
        for t in xrange(self.T):
            self.update_theta_batch(t, disp)
            if verbose and not t % 100:
                sys.stdout.write('.')
        if verbose:
            t = time.time() - start_t
            sys.stdout.write('\n')
            print 'Batch update\ttime: {:.2f}'.format(t)
            obj = self.bound()
            diff_str = '+' if obj > last_score else '-'
            print('Update (A)\tBefore: {:.2f}\tAfter: {:.2f}\t{}'.format(
                last_score, obj, diff_str))

    def update_theta_batch(self, t, disp):
        def f(theta):
            a, b = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Eloga = comp_expect(a, b)
            Eexp = self.comp_exp_expect(a[:, np.newaxis],
                                        b[:, np.newaxis],
                                        self.U)

            likeli = (-self.W[t] * np.prod(Eexp, axis=0)
                      - np.dot(Ea, self.U)) * self.gamma
            prior = (self.alpha - 1) * Eloga - self.alpha * Ea
            ent = entropy(a, b)

            return -(likeli.sum() + prior.sum() + ent.sum())

        def df(theta):
            a, b = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, _ = comp_expect(a, b)
            Eexp = self.comp_exp_expect(a[:, np.newaxis],
                                        b[:, np.newaxis],
                                        self.U)

            tmp = self.U / b[:, np.newaxis]
            log_term, inv_term = np.empty_like(tmp), np.empty_like(tmp)
            idx = (tmp > -1)
            # log(1 + x) is better approximated as x if x is sufficiently small
            idx_napp = np.logical_and(idx, np.abs(tmp) > 1e-15)
            idx_app = (np.abs(tmp) <= 1e-15)
            log_term[idx_napp] = np.log(1. + tmp[idx_napp])
            log_term[idx_app] = tmp[idx_app]
            log_term[-idx] = -np.inf
            inv_term[idx], inv_term[-idx] = 1. / (1. + tmp[idx]), np.inf

            grad_a = np.sum(self.W[t] * log_term * np.prod(Eexp, axis=0) *
                            self.gamma - self.U / b[:, np.newaxis] *
                            self.gamma, axis=1)
            grad_a = grad_a + (self.alpha - a) * special.polygamma(1, a)
            grad_a = grad_a + 1 - self.alpha / b
            grad_b = a/b**2 * np.sum(-self.U * self.W[t] * inv_term *
                                     np.prod(Eexp, axis=0) * self.gamma +
                                     self.U * self.gamma, axis=1)
            grad_b = grad_b + self.alpha * (a/b**2 - 1./b)
            return -np.hstack((a * grad_a, b * grad_b))

        theta0 = np.hstack((np.log(self.a[t]), np.log(self.b[t])))
        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'A[{}, :]: {}, f={}'.format(t,
                                                  d['task'],
                                                  f(theta_hat))
            else:
                print 'A[{}, :]: {}, f={}'.format(t,
                                                  d['warnflag'],
                                                  f(theta_hat))
            app_grad = approx_grad(f, theta_hat)
            ana_grad = df(theta_hat)
            for l in xrange(self.L):
                print_gradient('log_a[{}, {:3d}]'.format(t, l),
                               theta_hat[l],
                               ana_grad[l],
                               app_grad[l])
                print_gradient('log_b[{}, {:3d}]'.format(t, l),
                               theta_hat[l + self.L],
                               ana_grad[l + self.L],
                               app_grad[l + self.L])

        self.a[t], self.b[t] = np.exp(theta_hat[:self.L]), np.exp(
            theta_hat[-self.L:])
        assert(np.all(self.a[t] > 0))
        assert(np.all(self.b[t] > 0))
        self.EA[t], self.ElogA[t] = comp_expect(self.a[t], self.b[t])

    def vb_m(self, batch=False, verbose=True, disp=0, update_alpha=True):
        """ Perform one M-step, update the model parameters with A fixed
        from E-step

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
        if verbose:
            old_U = self.U.copy()
            old_gamma = self.gamma.copy()
            old_alpha = self.alpha.copy()
            last_score = self.bound()
            print('Update (initial)\tObj: {:.2f}'.format(last_score))
            start_t = time.time()
        if batch:
            self.update_u_batch(disp)
            if verbose:
                obj = self.bound()
                diff_str = '+' if obj > last_score else '-'
                print('Update (U)\tBefore: {:.2f}\tAfter: {:.2f}\t{}'.format(
                    last_score,
                    obj,
                    diff_str))
                last_score = obj
        else:
            for l in xrange(self.L):
                self.update_u(l, disp)
                if verbose:
                    obj = self.bound()
                    diff_str = '+' if obj > last_score else '-'
                    print('Update (U[{}])\tBefore: {:.2f}\tAfter:'
                          ' {:.2f}\t{}'.format(l, last_score, obj, diff_str))
                    last_score = obj
        self.update_gamma(disp)
        if verbose:
            obj = self.bound()
            diff_str = '+' if obj > last_score else '-'
            print('Update (gamma)\tBefore: {:.2f}\tAfter:'
                  ' {:.2f}\t{}'.format(last_score, obj, diff_str))
            last_score = obj

        if update_alpha:
            self.update_alpha(disp)
            if verbose:
                obj = self.bound()
                diff_str = '+' if obj > last_score else '-'
                print('Update (alpha)\tBefore: {:.2f}\tAfter:'
                      ' {:.2f}\t{}'.format(last_score, obj, diff_str))

        if verbose:
            t = time.time() - start_t
            U_diff = np.mean(np.abs(self.U - old_U))
            sigma_diff = np.mean(np.abs(np.sqrt(1. / self.gamma) -
                                        np.sqrt(1. / old_gamma)))
            alpha_diff = np.mean(np.abs(self.alpha - old_alpha))
            print('U diff: {:.4f}\tsigma dff: {:.4f}\talpha diff: {:.4f}\t'
                  'time: {:.2f}'.format(U_diff, sigma_diff, alpha_diff, t))

    def update_u_batch(self, disp):
        def f_df(u):
            U = u.reshape(self.L, self.F)
            Eexp = 1.
            for l in xrange(self.L):
                Eexp = Eexp * self.comp_exp_expect(self.a[:, l, np.newaxis],
                                                   self.b[:, l, np.newaxis],
                                                   U[l, :])
            grad_U = np.zeros_like(U)
            for l in xrange(self.L):
                tmp = 1 + U[l] / self.b[:, l, np.newaxis]
                inv_term = np.empty_like(tmp)
                idx = (tmp > 0)
                inv_term[idx], inv_term[-idx] = 1. / tmp[idx], np.inf
                grad_U[l] = np.sum(self.EA[:, l, np.newaxis] *
                                   (1 - self.W * Eexp * inv_term))
            return (np.sum(np.dot(self.EA, U) + self.W * Eexp), grad_U.ravel())

        u0 = self.U.ravel()
        u_hat, _, d = optimize.fmin_l_bfgs_b(f_df, u0, disp=0)
        self.U = u_hat.reshape(self.L, self.F)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'U: {}, f={}'.format(d['task'], f_df(u_hat)[0])
            else:
                print 'U: {}, f={}'.format(d['warnflag'], f_df(u_hat)[0])

    def update_u(self, l, disp):
        def f(u):
            Eexp = self.comp_exp_expect(self.a[:, l, np.newaxis],
                                        self.b[:, l, np.newaxis],
                                        u)
            return np.sum(np.outer(self.EA[:, l], u) + self.W * Eexp * Eres)

        def df(u):
            tmp = self.comp_exp_expect(self.a[:, l, np.newaxis] + 1,
                                       self.b[:, l, np.newaxis],
                                       u)
            return np.sum(self.EA[:, l, np.newaxis] *
                          (1 - self.W * Eres * tmp), axis=0)

        k_idx = np.delete(np.arange(self.L), l)
        Eres = 1.
        for k in k_idx:
            Eres = Eres * self.comp_exp_expect(self.a[:, k, np.newaxis],
                                               self.b[:, k, np.newaxis],
                                               self.U[k])

        u0 = self.U[l]
        self.U[l], _, d = optimize.fmin_l_bfgs_b(f, u0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'U[{}, :]: {}, f={}'.format(l,
                                                  d['task'],
                                                  f(self.U[l]))
            else:
                print 'U[{}, :]: {}, f={}'.format(l,
                                                  d['warnflag'],
                                                  f(self.U[l]))
            app_grad = approx_grad(f, self.U[l])
            ana_grad = df(self.U[l])
            for fr in xrange(self.F):
                print_gradient('U[{}, {:3d}]'.format(l, fr), self.U[l, fr],
                               ana_grad[fr], app_grad[fr])

    def update_gamma(self, disp):
        def f(eta):
            gamma = np.exp(eta)
            return -(self.T * np.sum(gamma * eta - special.gammaln(gamma)) +
                     np.sum(gamma * np.log(self.W) - gamma *
                            np.dot(self.EA, self.U) - gamma * self.W * Eexp))

        def df(eta):
            gamma = np.exp(eta)
            return -gamma * (self.T * (eta + 1 - special.psi(gamma)) +
                             np.sum(-np.dot(self.EA, self.U) +
                                    np.log(self.W) - self.W * Eexp, axis=0))

        Eexp = 1.
        for l in xrange(self.L):
            Eexp = Eexp * self.comp_exp_expect(self.a[:, l, np.newaxis],
                                               self.b[:, l, np.newaxis],
                                               self.U[l])

        eta0 = np.log(self.gamma)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.gamma = np.exp(eta_hat)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(f(eta_hat), d['task'])
            else:
                print 'f={}, {}'.format(f(eta_hat), d['warnflag'])
            app_grad = approx_grad(f, eta_hat)
            ana_grad = df(eta_hat)
            for idx in xrange(self.F):
                print_gradient('Gamma[{:3d}]'.format(idx), self.gamma[idx],
                               ana_grad[idx], app_grad[idx])

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
                print 'f={}, {}'.format(f(eta_hat), d['task'])
            else:
                print 'f={}, {}'.format(f(eta_hat), d['warnflag'])
            app_grad = approx_grad(f, eta_hat)
            ana_grad = df(eta_hat)
            for l in xrange(self.L):
                print_gradient('Alpha[{:3d}]'.format(l), self.alpha[l],
                               ana_grad[l], app_grad[l])

    def bound(self):
        Eexp = 1.
        for l in xrange(self.L):
            Eexp = Eexp * self.comp_exp_expect(self.a[:, l, np.newaxis],
                                               self.b[:, l, np.newaxis],
                                               self.U[l])
        # E[log P(w|a)]
        bound = self.T * np.sum(self.gamma * np.log(self.gamma) -
                                special.gammaln(self.gamma))
        bound = bound + np.sum(-self.gamma * np.dot(self.EA, self.U) +
                               (self.gamma - 1) * np.log(self.W) -
                               self.W * Eexp * self.gamma)
        # E[log P(a)]
        bound = bound + self.T * np.sum(self.alpha * np.log(self.alpha) -
                                        special.gammaln(self.alpha))
        bound = bound + np.sum(self.ElogA * (self.alpha - 1) -
                               self.EA * self.alpha)
        # E[loq q(a)]
        bound = bound + np.sum(entropy(self.a, self.b))
        return bound

    def comp_exp_expect(self, alpha, beta, U):
        ''' Compute E[exp(-au)] where a ~ Gamma(alpha, beta) and u constant

        This function makes extensive use of broadcasting, thus the dimension
        of input arguments can only be one of the following two situations:
             1) U has shape (L, F), alpha and beta have shape (L, 1)
                --> output shape (L, F)
             2) U has shape (F, ), alpha and beta have shape (T, 1)
                --> output shape (T, F)
        '''
        # using Taylor expansion for large alpha (hence beta) to more
        # accurately compute (1 + u/beta)**(-alpha)
        idx = np.logical_and(alpha < 1e10, beta < 1e10).ravel()
        if alpha.size == self.L:
            expect = np.empty_like(U)
            expect[idx] = (1 + U[idx] / beta[idx])**(-alpha[idx])
            expect[-idx] = np.exp(-U[-idx] * alpha[-idx] / beta[-idx])
        elif alpha.size == self.T:
            expect = np.empty((self.T, self.F))
            expect[idx] = (1 + U / beta[idx])**(-alpha[idx])
            expect[-idx] = np.exp(-U * alpha[-idx] / beta[-idx])
        else:
            raise ValueError('wrong dimension')
        expect[U <= -beta] = np.inf
        return expect


def print_gradient(name, val, grad, approx):
    print('{} = {:.2f}\tGradient: {:.2f}\tApprox: {:.2f}\t'
          '| Diff |: {:.3f}'.format(name, val, grad, approx,
                                    np.abs(grad - approx)))


def comp_expect(alpha, beta):
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def entropy(alpha, beta):
    return (alpha - np.log(beta) + special.gammaln(alpha) +
            (1 - alpha) * special.psi(alpha))


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2 * delta)
    return grad
