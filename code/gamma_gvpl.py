"""

Product-of-Filter model with gamma noise model

CREATED: 2013-07-12 11:09:44 by Dawen Liang <daliang@adobe.com>

"""

import time
import sys

import numpy as np

from scipy import optimize, special, weave


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

    def vb_e(self, cold_start=False, smoothness=100, verbose=True, disp=0):
        """ Perform one variational E-step to appxorimate the posterior
        P(A | -)

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
            #last_score = self.bound()
            self.update_theta(t, disp)
            #score = self.bound()
            #if score < last_score:
            #    print('Oops, before: {}\tafter: {}\tt={}'.format(
            #        last_score, score, t))
            if verbose and not t % 100:
                sys.stdout.write('.')
        if verbose:
            t = time.time() - start_t
            sys.stdout.write('\n')
            print 'Batch update\ttime: {:.2f}'.format(t)
            score = self.bound()
            print_increment('A', last_score, score)

    def update_theta(self, t, disp):
        def f(theta):
            a, b = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Eloga = comp_expect(a, b)
            logEexp = _comp_logEexp(a, b, self.U, update_U=False)
            likeli = (-self.W[t] * np.exp(np.sum(logEexp, axis=0))
                      - np.dot(Ea, self.U)) * self.gamma
            prior = (self.alpha - 1) * Eloga - self.alpha * Ea
            ent = entropy(a, b)

            return -(likeli.sum() + prior.sum() + ent.sum())

        def df(theta):
            a, b = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            logEexp = _comp_logEexp(a, b, self.U, update_U=False)

            tmp = self.U / b[:, np.newaxis]
            log_term, inv_term = np.empty_like(tmp), np.empty_like(tmp)
            idx = (tmp > -1)
            log_term[idx] = np.log1p(tmp[idx])
            log_term[-idx] = -np.inf
            inv_term[idx], inv_term[-idx] = 1. / (1. + tmp[idx]), np.inf

            grad_a = np.sum(self.W[t] * log_term * np.exp(
                np.sum(logEexp, axis=0)) *
                self.gamma - self.U / b[:, np.newaxis] * self.gamma, axis=1)
            grad_a = grad_a + (self.alpha - a) * special.polygamma(1, a)
            grad_a = grad_a + 1 - self.alpha / b
            grad_b = a / b**2 * np.sum(-self.U * self.W[t] * inv_term *
                                       np.exp(np.sum(logEexp, axis=0)) *
                                       self.gamma + self.U * self.gamma,
                                       axis=1)
            grad_b = grad_b + self.alpha * (a / b**2 - 1. / b)
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

    def vb_m(self, verbose=True, disp=0):
        """ Perform one M-step, update the model parameters with A fixed
        from E-step

        Parameters
        ----------
        verbose: bool
            Output log if ture.
        disp: int
            Display warning from solver if > 0, mostly from LBFGS.
        """

        print 'Variational M-step...'
        if verbose:
            old_U = self.U.copy()
            old_gamma = self.gamma.copy()
            old_alpha = self.alpha.copy()
            last_score = self.bound()
            print('Update (initial)\tObj: {:.2f}'.format(last_score))
            start_t = time.time()
        for f in xrange(self.F):
            self.update_u(f, disp)
            #score = self.bound()
            #if score < last_score:
            #    print('Oops, before: {}\tafter: {}\tf={}'.format(
            #        last_score, score, f))
            #last_score = score
        self.update_gamma(disp)
        if verbose:
            score = self.bound()
            print_increment('gamma', last_score, score)
            last_score = score

        self.update_alpha(disp)
        if verbose:
            score = self.bound()
            print_increment('alpha', last_score, score)

        if verbose:
            t = time.time() - start_t
            U_diff = np.mean(np.abs(self.U - old_U))
            sigma_diff = np.mean(np.abs(np.sqrt(1. / self.gamma) -
                                        np.sqrt(1. / old_gamma)))
            alpha_diff = np.mean(np.abs(self.alpha - old_alpha))
            print('U diff: {:.4f}\tsigma dff: {:.4f}\talpha diff: {:.4f}\t'
                  'time: {:.2f}'.format(U_diff, sigma_diff, alpha_diff, t))

    def update_u(self, f, disp):
        def fun(u):
            Eexp = np.exp(np.sum(_comp_logEexp(self.a, self.b, u), axis=1))
            return np.sum(self.gamma[f] * (Eexp * self.W[:, f] +
                                           np.dot(self.EA, u)))

        def dfun(u):
            tmp = 1 + u / self.b
            inv_term = np.empty_like(tmp)
            idx = (tmp > 0)
            inv_term[idx], inv_term[-idx] = 1. / tmp[idx], np.inf
            Eexp = np.exp(np.sum(_comp_logEexp(self.a, self.b, u), axis=1))
            return np.sum(self.EA * (1 - (self.W[:, f] * Eexp)[:, np.newaxis] *
                                     inv_term), axis=0)

        u0 = self.U[:, f]
        self.U[:, f], _, d = optimize.fmin_l_bfgs_b(fun, u0, fprime=dfun, disp=0)
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

        Eexp = np.exp(comp_logEexp(self.a, self.b, self.U))

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
        Eexp = np.exp(comp_logEexp(self.a, self.b, self.U))
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


def print_gradient(name, val, grad, approx):
    print('{} = {:.2f}\tGradient: {:.2f}\tApprox: {:.2f}\t'
          '| Diff |: {:.3f}'.format(name, val, grad, approx,
                                    np.abs(grad - approx)))


def print_increment(name, last_score, score):
    diff_str = '+' if score > last_score else '-'
    print('Update ({})\tBefore: {:.2f}\tAfter: {:.2f}\t{}'.format(
        name, last_score, score, diff_str))


def comp_expect(alpha, beta):
    return (alpha / beta, special.psi(alpha) - np.log(beta))


def entropy(alpha, beta):
    ''' Compute the entropy of a r.v. theta ~ Gamma(alpha, beta)
    '''
    return (alpha - np.log(beta) + special.gammaln(alpha) +
            (1 - alpha) * special.psi(alpha))


def _comp_logEexp(a, b, U, update_U=True):
    if update_U:
        # a, b: (T, L)      U: (L, )
        #   --> output: (T, L)
        T, L = a.shape
        log_exp = np.empty((T, L))
        comp_expectation = r"""
        int t, l;
        for (t = 0; t < T; t++) {
            for (l = 0; l < L; l++) {
                if (U[l] / b[(t*L) + l] > -1) {
                    log_exp[(t*L) + l] = -a[(t*L) + l] * log1p(U[l] / b[(t*L) + l]);
                } else {
                    log_exp[(t*L) + l] = INFINITY;
                }
            }
        }
        """
        weave.inline(comp_expectation,
                     ['T', 'L', 'a', 'b', 'U', 'log_exp'])

    else:
        # a, b: (L, )       U: (L, F)
        #   --> output: (L, F)
        L, F = U.shape
        log_exp = np.empty((L, F))
        comp_expectation = r"""
        int l, f;
        for (l = 0; l < L; l++) {
            for (f = 0; f < F; f++) {
                if (U[(l*F) + f] / b[l] > -1) {
                    log_exp[(l*F) + f] = -a[l] * log1p(U[(l*F) + f]/b[l]);
                } else {
                    log_exp[(l*F) + f] = INFINITY;
                }
            }
        }
        """
        weave.inline(comp_expectation,
                     ['L', 'F', 'a', 'b', 'U', 'log_exp'])

    return log_exp


def comp_logEexp(a, b, U):
    # log(E(\prod_l exp(U_{fl} a_{lt})))
    T, L = a.shape
    F = U.shape[1]
    log_exp = np.empty((T, F))
    comp_expectation = r"""
        int t, f, l;
        for (t = 0; t < T; t++) {
            for (f = 0; f < F; f++) {
                 log_exp[(t*F) + f] = 0.0;
                 for (l = 0; l < L; l++) {
                    if (U[(l*F) + f] / b[(t*L) + l] > -1) {
                        log_exp[(t*F) + f] += -a[(t*L) + l] * log1p(U[(l*F) + f] / b[(t*L) + l]);
                    } else {
                        log_exp[(t*F) + f] = INFINITY;
                    }
                 }
             }
         }
    """
    weave.inline(comp_expectation,
                 ['T', 'L', 'F', 'a', 'b', 'U', 'log_exp'])
    return log_exp


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2 * delta)
    return grad
