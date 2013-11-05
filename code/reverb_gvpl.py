"""

Decoloration model based PoF

CREATED: 2013-07-12 11:09:44 by Dawen Liang <daliang@adobe.com>

"""

import time
import sys

import numpy as np
import scipy.optimize as optimize
import scipy.special as special


class SF_Dict(object):
    def __init__(self, W, U, alpha, gamma, L=10, smoothness=100,
                 flat_init=True, seed=None):
        self.W = W.copy()
        self.T, self.F = W.shape
        self.L = L
        if seed is None:
            print 'Using random seed'
            np.random.seed()
        else:
            print 'Using fixed seed {}'.format(seed)
            np.random.seed(seed)
        self.U, self.alpha, self.gamma = U, alpha, gamma
        self.reverb = np.random.randn(self.F)
        self._init_variational(smoothness, flat_init)

    def _init_variational(self, smoothness, flat_init):
        if flat_init:
            self.a = smoothness * np.random.gamma(smoothness,
                                                  1. / smoothness,
                                                  size=(self.T, self.L))
            self.b = smoothness * np.random.gamma(smoothness,
                                                  1. / smoothness,
                                                  size=(self.T, self.L))
        else:
            EA = np.exp(special.psi(self.alpha) - np.log(self.alpha))
            self.a = smoothness * np.ones((self.T, self.L))
            self.b = smoothness * np.ones((self.T, self.L)) / EA
        self.EA, self.ElogA = comp_expect(self.a, self.b)

    def vb_e(self, cold_start=False, smoothness=100, maxiter=15000,
             verbose=True, disp=0):
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
            self.update_theta_batch(t, maxiter, disp)
            #score = self.bound()
            #if score < last_score:
            #    print('Oops, before: {}\tafter: {}\tt={}'.format(
            #        last_score, score, t))
            if verbose and not t % 20:
                sys.stdout.write('.')
        if verbose:
            t = time.time() - start_t
            sys.stdout.write('\n')
            print 'Batch update\ttime: {:.2f}'.format(t)
            score = self.bound()
            print_increment('A', last_score, score)

    def update_theta_batch(self, t, maxiter, disp):
        def f(theta):
            a, b = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Eloga = comp_expect(a, b)
            logEexp = comp_log_exp(a[:, np.newaxis],
                                   b[:, np.newaxis], self.U)
            likeli = -(self.W[t] * np.exp(np.sum(logEexp, axis=0) - self.reverb)
                       + np.dot(Ea, self.U)) * self.gamma
            prior = (self.alpha - 1) * Eloga - self.alpha * Ea
            ent = entropy(a, b)

            return -(likeli.sum() + prior.sum() + ent.sum())

        def df(theta):
            a, b = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            logEexp = comp_log_exp(a[:, np.newaxis],
                                   b[:, np.newaxis], self.U)

            tmp = self.U / b[:, np.newaxis]
            log_term, inv_term = np.empty_like(tmp), np.empty_like(tmp)
            idx = (tmp > -1)
            # log(1 + x) is better approximated as x if x is sufficiently small
            idx_dir = np.logical_and(idx, np.abs(tmp) > 1e-12)
            idx_app = (np.abs(tmp) <= 1e-12)
            log_term[idx_dir] = np.log(1. + tmp[idx_dir])
            log_term[idx_app] = tmp[idx_app]
            log_term[-idx] = -np.inf
            inv_term[idx], inv_term[-idx] = 1. / (1. + tmp[idx]), np.inf

            grad_a = np.sum(self.W[t] * np.exp(-self.reverb) * log_term *
                            np.exp(np.sum(logEexp, axis=0)) * self.gamma -
                            self.U / b[:, np.newaxis] * self.gamma, axis=1)
            grad_a = grad_a + (self.alpha - a) * special.polygamma(1, a)
            grad_a = grad_a + 1 - self.alpha / b
            grad_b = a / b**2 * np.sum(-self.U * self.W[t] * np.exp(-self.reverb) * inv_term *
                                       np.exp(np.sum(logEexp, axis=0)) *
                                       self.gamma + self.U * self.gamma,
                                       axis=1)
            grad_b = grad_b + self.alpha * (a / b**2 - 1. / b)
            return -np.hstack((a * grad_a, b * grad_b))

        theta0 = np.hstack((np.log(self.a[t]), np.log(self.b[t])))
        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df,
                                                 maxiter=maxiter, disp=0)
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

    def vb_m(self, batch=False, maxiter=15000, verbose=True, disp=0):
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
            old_reverb = self.reverb.copy()
            last_score = self.bound()
            print('Update (initial)\tObj: {:.2f}'.format(last_score))
            start_t = time.time()
        if batch:
            for f in xrange(self.F):
                self.update_reverb_batch(f, maxiter, disp)
                score = self.bound()
                if score < last_score:
                    print('Oops, before: {}\tafter: {}\tf={}'.format(
                        last_score, score, f))
                last_score = score
        else:
            self.update_reverb()
            if verbose:
                score = self.bound()
                print_increment('reverb', last_score, score)
                last_score = score
        if verbose:
            t = time.time() - start_t
            reverb_diff = np.mean(np.abs(self.reverb - old_reverb))
            print('reverb diff: {:.4f}\ttime: {:.2f}'.format(reverb_diff, t))

    def update_reverb(self):
        Eres = 0.
        for l in xrange(self.L):
            Eres = Eres + comp_log_exp(self.a[:, l, np.newaxis],
                                       self.b[:, l, np.newaxis],
                                       self.U[l])
        Eres = np.exp(Eres)
        self.reverb = np.log(np.mean(self.W * Eres, axis=0))

    def bound(self):
        Eexp = 0.
        for l in xrange(self.L):
            Eexp = Eexp + comp_log_exp(self.a[:, l, np.newaxis],
                                       self.b[:, l, np.newaxis],
                                       self.U[l])
        Eexp = np.exp(Eexp)
        # E[log P(w|a)]
        bound = self.T * np.sum(self.gamma * np.log(self.gamma) -
                                special.gammaln(self.gamma))
        bound = bound + np.sum(-self.gamma * (np.dot(self.EA, self.U) + self.reverb) +
                               (self.gamma - 1) * np.log(self.W) -
                               self.W * np.exp(-self.reverb) * Eexp * self.gamma)
        # E[log P(a)]
        bound = bound + self.T * np.sum(self.alpha * np.log(self.alpha) -
                                        special.gammaln(self.alpha))
        bound = bound + np.sum(self.ElogA * (self.alpha - 1) -
                               self.EA * self.alpha)
        # E[loq q(a)]
        bound = bound + np.sum(entropy(self.a, self.b))
        return bound

    ## This function is deprecated
    #def comp_exp_expect(self, alpha, beta, U):
    #    ''' Compute E[exp(-au)] where a ~ Gamma(alpha, beta) and u constant

    #    This function makes extensive use of broadcasting, thus the dimension
    #    of input arguments can only be one of the following two situations:
    #         1) U has shape (L, F), alpha and beta have shape (L, 1)
    #            --> output shape (L, F)
    #         2) U has shape (F, ), alpha and beta have shape (T, 1)
    #            --> output shape (T, F)
    #    '''
    #    # using Taylor expansion for large alpha (hence beta) to more
    #    # accurately compute (1 + u/beta)**(-alpha)
    #    idx = np.logical_and(alpha < 1e10, beta < 1e10).ravel()
    #    if alpha.size == self.L:
    #        expect = np.empty_like(U)
    #        expect[idx] = (1 + U[idx] / beta[idx])**(-alpha[idx])
    #        expect[-idx] = np.exp(-U[-idx] * alpha[-idx] / beta[-idx])
    #    elif alpha.size == self.T:
    #        expect = np.empty((self.T, self.F))
    #        expect[idx] = (1 + U / beta[idx])**(-alpha[idx])
    #        expect[-idx] = np.exp(-U * alpha[-idx] / beta[-idx])
    #    else:
    #        raise ValueError('wrong dimension')
    #    expect[U <= -beta] = np.inf
    #    return expect


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


def comp_log_exp(alpha, beta, U):
    ''' Compute log(E[exp(-au)]) where a ~ Gamma(alpha, beta) and u as a
    constant:
        log(E[exp(-au)]) = -alpha * log(1 + u / beta)
    Like self.comp_exp_expect (deprecated), this function makes extensive
    use of broadcasting. Therefore, the dimension of the input arguments
    (at least by design) can only be one of the following two situations:
        1) U: (L, F)    alpha, beta: (L, 1)
            --> output: (L, F)
        2) U: (F, )     alpha, beta: (T, 1)
            --> oupput: (T, F)
    '''
    tmp = U / beta
    log_exp = np.empty_like(tmp)
    idx = (tmp > -1)
    # log(1 + x) is better approximated as x if x is sufficiently small
    # otherwise, it can be directly computed
    idx_dir = np.logical_and(idx, np.abs(tmp) > 1e-12)
    idx_app = (np.abs(tmp) <= 1e-12)
    log_exp[idx_dir] = (-alpha * np.log(1. + tmp))[idx_dir]
    log_exp[idx_app] = (-alpha * tmp)[idx_app]
    log_exp[-idx] = np.inf
    return log_exp


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2 * delta)
    return grad
