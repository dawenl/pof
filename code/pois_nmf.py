"""
Poisson NMF

CREATED: 2013-08-09 23:50:58 by Dawen Liang <dl2771@columbia.edu>

"""

import functools
from math import log

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as optimize
import scipy.special as special

import utils

specshow = functools.partial(plt.imshow, cmap=plt.cm.jet, origin='lower',
                             aspect='auto', interpolation='nearest')


class KL_NMF:
    def __init__(self, X, K=100, d=100, GaP=False, smoothness=100, seed=None,
                 **kwargs):
        self.X = X.copy()
        self.K, self.d = K, d
        self.GaP = GaP
        self.F, self.T = X.shape
        if seed is None:
            print 'Using random seed'
            np.random.seed()
        else:
            print 'Using fixed seed {}'.format(seed)
            np.random.seed(seed)
        self._parse_hyperparameter(**kwargs)
        self._init(smoothness)

    def _parse_hyperparameter(self, **kwargs):
        self.a = float(kwargs['a']) if 'a' in kwargs else 0.1
        self.b = float(kwargs['b']) if 'b' in kwargs else 0.1
        self.beta = float(kwargs['beta']) if 'beta' in kwargs else 1.
        # source-filter prior
        if 'U' in kwargs:
            self.U = kwargs['U'].copy()
            self.sf_prior = True
        else:
            self.U = None
            self.sf_prior = False
        self.alpha = kwargs['alpha'].copy() if 'alpha' in kwargs else None
        self.gamma = kwargs['gamma'].copy() if 'gamma' in kwargs else None
        if self.gamma is not None and self.gamma.ndim == 1:
            self.gamma = self.gamma[:, np.newaxis]

    def _init(self, smoothness):
        if self.sf_prior:
            self.nua = 10000 * np.random.gamma(smoothness,
                                               1. / smoothness,
                                               size=(self.L, self.K))
            self.rhoa = 10000 * np.random.gamma(smoothness,
                                                1. / smoothness,
                                                size=(self.L, self.K))

        self.nuw = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                           size=(self.F, self.K))
        self.rhow = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                            size=(self.F, self.K))
        self.nuh = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                           size=(self.F, self.K))
        self.rhoh = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                            size=(self.K, self.T))
        if self.GaP:
            self.nut = 10000 * np.random.gamma(smoothness, 1. / smoothness,
                                               size=(self.K, ))
            self.rhot = self.K * 10000 * np.random.gamma(smoothness,
                                                         1. / smoothness,
                                                         size=(self.K, ))
        self.compute_expectations()

    def compute_expectations(self):
        if self.sf_prior:
            self.Ea, self.Eloga = utils.compute_gamma_expectation(self.nua,
                                                                  self.rhoa)
        self.Ew, self.Elogw = utils.compute_gamma_expectation(self.nuw,
                                                              self.rhow)
        self.Eh, self.Elogh = utils.compute_gamma_expectation(self.nuh,
                                                              self.rhoh)
        if self.GaP:
            self.Et, self.Elogt = utils.compute_gamma_expectation(self.nut,
                                                                  self.rhot)
        else:
            self.Et, self.Elogt = np.ones((self.K, )), self.zeros((self.K, ))

    def update(self, disp):
        self.update_h()
        self.update_w()
        goodk = self.goodk()
        if self.sf_prior:
            for k in goodk:
                self.update_a(k, disp)
        if self.GaP:
            self.update_theta()
            # truncate unused components
            self.clear_badk()

    def update_a(self, k, disp):
        def f(theta):
            nu, rho = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Eloga = utils.compute_gamma_expectation(nu, rho)
            # E[p(W|a)] + E[p(a)] - E[q(a)], collect terms by sufficient
            # statistics
            val = np.sum((self.alpha - nu) * Eloga)
            val = val - np.sum((self.alpha - rho) * Ea)
            val = val + np.sum(special.gammaln(nu) - nu * np.log(rho))

            logEexp = utils.comp_log_exp(nu, rho, self.U)
            val = val - np.sum(self.gamma.ravel() * (self.Ew[:, k] *
                               np.exp(np.sum(logEexp, axis=1)) +
                               np.dot(self.U, Ea)))
            return -val

        def df(theta):
            nu, rho = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            logEexp = utils.comp_log_exp(nu, rho, self.U)

            tmp = self.U / rho
            log_term, inv_term = np.empty_like(tmp), np.empty_like(tmp)
            idx = (tmp > -1)
            # log(1 + x) is better approximated as x if x is sufficiently small
            idx_dir = np.logical_and(idx, np.abs(tmp) > 1e-12)
            idx_app = (np.abs(tmp) <= 1e-12)
            log_term[idx_dir] = np.log(1. + tmp[idx_dir])
            log_term[idx_app] = tmp[idx_app]
            log_term[-idx] = -np.inf
            inv_term[idx], inv_term[-idx] = 1. / (1. + tmp[idx]), np.inf

            grad_nu = np.sum((self.Ew[:, k, np.newaxis] * log_term *
                              np.exp(np.sum(logEexp, axis=1, keepdims=True))
                              - self.U / rho) * self.gamma, axis=0)
            grad_nu = grad_nu + 1 - self.alpha / rho
            grad_nu = grad_nu + (self.alpha - nu) * special.polygamma(1, nu)

            grad_rho = np.sum((self.U - self.Ew[:, k, np.newaxis] *
                               self.U * inv_term *
                               np.exp(np.sum(logEexp, axis=1, keepdims=True)))
                              * self.gamma, axis=0)
            grad_rho = nu / rho**2 * grad_rho
            grad_rho = grad_rho + self.alpha * (nu / rho**2 - 1. / rho)
            return -np.hstack((nu * grad_nu, rho * grad_rho))

        theta0 = np.hstack((np.log(self.nua[:, k]), np.log(self.rhoa[:, k])))
        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'A[:, {}]: {}, f={}'.format(k, d['task'],
                                                  f(theta_hat))
            else:
                print 'A[:, {}]: {}, f={}'.format(k, d['warnflag'],
                                                  f(theta_hat))
            app_grad = approx_grad(f, theta_hat)
            ana_grad = df(theta_hat)
            for l in xrange(self.L):
                if abs(ana_grad[l] - app_grad[l]) > .05:
                    print_gradient('log_a[{}, {:3d}]'.format(l, k),
                                   theta_hat[l], ana_grad[l], app_grad[l])
                    print_gradient('log_b[{}, {:3d}]'.format(l, k),
                                   theta_hat[l + self.L], ana_grad[l + self.L],
                                   app_grad[l + self.L])

        self.nua[:, k], self.rhoa[:, k] = np.exp(theta_hat[:self.L]), np.exp(
            theta_hat[-self.L:])
        assert(np.all(self.nua[:, k] > 0))
        assert(np.all(self.rhoa[:, k] > 0))
        self.Ea[:, k], self.Eloga[:, k] = utils.compute_gamma_expectation(
            self.nua[:, k], self.rhoa[:, k])
        self.logEexpa[:, :, k] = utils.comp_log_exp(self.nua[:, k],
                                                    self.rhoa[:, k],
                                                    self.U)

    def update_w(self):
        goodk = self.goodk()
        c = self.X.sum() / self._xbar().sum()

        xxelinv = self.X / self._xexplog(goodk)

        if self.sf_prior:
            self.nuw[:, goodk] = self.gamma
            self.nuw[:, goodk] = self.nuw[:, goodk] + self.d * np.exp(self.Elogw[:, goodk] + self.Elogt[goodk]) * np.dot(xxelinv, np.exp(self.Elogh[goodk]).T)
            self.rhow[:, goodk] = self.gamma * np.exp(np.sum(self.logEexpa[:, :, goodk], axis=1))
            self.rhow[:, goodk] = self.rhow[:, goodk] + c * self.d * np.sum(self.Eh[goodk], axis=1, keepdims=True).T
        else:
            self.nuw[:, goodk] = self.a + self.d * np.exp(self.Elogw[:, goodk]
                                                          + self.Elogt[goodk]) * np.dot(xxelinv, np.exp(self.Elogh[goodk]).T)
            self.rhow[:, goodk] = self.a + c * self.d * np.sum(self.Eh[goodk],
                                                               axis=1, keepdims=True).T
        self.Ew[:, goodk], self.Elogw[:, goodk] = utils.compute_gamma_expectation(
            self.nuw[:, goodk], self.rhow[:, goodk]
        )

    def update_h(self):
        goodk = self.goodk()
        c = self.X.sum() / self._xbar().sum()

        xxelinv = self.X / self._xexplog(goodk)

        self.nuh[goodk] = self.b + self.d * np.exp(self.Elogh[goodk, :]) * np.dot(np.exp(self.Elogt[goodk] + self.Elogw[:, goodk]).T, xxelinv)
        self.rhoh[goodk] = self.b + c * self.d * np.sum(self.Ew[:, goodk], axis=0, keepdims=True).T
        self.Eh[goodk], self.Elogh[goodk] = utils.compute_gamma_expectation(
            self.nuh[goodk], self.rhoh[goodk])

    def update_theta(self):
        goodk = self.goodk()
        c = self.X.sum() / self._xbar().sum()

        xxelinv = self.X / self._xexplog(goodk)

        self.nut[goodk] = self.beta / self.K
        self.nut[goodk] = self.nut[goodk] + self.d * np.exp(self.Elogt[goodk]) * np.sum(np.exp(self.Elogh[goodk] * np.dot(np.exp(self.Elogw[:, goodk]).T, xxelinv)), axis=1)
        self.rhot[goodk] = self.beta + c * self.d * np.sum(self.Ew, axis=0) * np.sum(self.Eh, axis=1)
        self.Et[goodk], self.Elogt[goodk] = utils.compute_gamma_expectation(
            self.nut[goodk], self.rhot[goodk]
        )
        pass

    def goodk(self):
        if not self.GaP:
            return np.arange(self.K)
        pass

    def clear_badk(self):
        pass

    def bound(self):
        score = 0
        goodk = self.goodk()
        c = self.X.sum() / self._xbar().sum()

        score = score + np.sum(self.d * self.X * (log(c * self.d) +
                                                  np.log(self._xexplog(goodk)))
                               - c * self.d * self._xbar(goodk))
        score = score + utils.gamma_term(self.Ew, self.Elogw, self.nuw,
                                         self.rhow)
        score = score + utils.gamma_term(self.Eh, self.Elogh, self.nuh,
                                         self.rhoh)
        if self.GaP:
            score = score + utils.gamma_term(self.Et, self.Elogt, self.nut,
                                             self.rhot)
        if self.sf_prior:
            score = score + utils.gamma_term(self.Ea, self.Eloga, self.nua,
                                             self.rhoa, self.alpha)
        return score

    def _xbar(self, goodk=None):
        if goodk is None:
            goodk = np.arange(self.K)
        return np.dot(self.Ew[:, goodk] * self.Et[goodk],
                      self.Eh[goodk, :])

    def _xexplog(self, goodk):
        '''
        sum_k exp(E[log theta_k * W_{fk} * H_{kt}])
        '''
        return np.dot(np.exp(self.Elogw[:, goodk] + self.Elogt[goodk]),
                      np.exp(self.Elogh[goodk]))


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2 * delta)
    return grad


def print_gradient(name, val, grad, approx):
    print('{} = {:.2f}\tGradient: {:.2f}\tApprox: {:.2f}\t'
          '| Diff |: {:.3f}'.format(name, val, grad, approx,
                                    np.abs(grad - approx)))
