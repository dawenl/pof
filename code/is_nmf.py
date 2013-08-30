"""
IS-NMF

CREATED: 2013-08-09 23:50:58 by Dawen Liang <dl2771@columbia.edu>

"""

import functools
import time
from math import log

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as optimize
import scipy.special as special

import utils

specshow = functools.partial(plt.imshow, cmap=plt.cm.jet, origin='lower',
                             aspect='auto', interpolation='nearest')


class IS_NMF:
    def __init__(self, X, K=100, GaP=False, smoothness=100, seed=None,
                 **kwargs):
        self.X = X.copy()
        self.K = K
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
            self.L = self.U.shape[1]
            self.sf_prior = True
            print 'Using source-filter prior'
        else:
            self.U = None
            self.sf_prior = False
            print 'Using regular prior'
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
            self.logEexpa = np.zeros((self.F, self.L, self.K))

        self.rhow = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.F, self.K))
        self.tauw = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.F, self.K))
        self.rhoh = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.K, self.T))
        self.tauh = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.K, self.T))
        self.rhot = self.K * 10000 * np.random.gamma(smoothness,
                                                     1. / smoothness,
                                                     size=(self.K, ))
        self.taut = 1. / self.K * 10000 * np.random.gamma(smoothness,
                                                          1. / smoothness,
                                                          size=(self.K, ))
        if self.GaP:
            self.rhot = self.K * 10000 * np.random.gamma(smoothness,
                                                         1. / smoothness,
                                                         size=(self.K, ))
            self.taut = 1. / self.K * 10000 * np.random.gamma(smoothness,
                                                              1. / smoothness,
                                                              size=(self.K, ))
        self.compute_expectations()

    def compute_expectations(self):
        self.Ew, self.Ewinv = utils.compute_gig_expectations(self.gamma,
                                                             self.rhow,
                                                             self.tauw)
        self.Ewinvinv = 1. / self.Ewinv
        self.Eh, self.Ehinv = utils.compute_gig_expectations(self.b,
                                                             self.rhoh,
                                                             self.tauh)
        self.Ehinvinv = 1. / self.Ehinv
        if self.GaP:
            self.Et, self.Etinv = utils.compute_gig_expectations(
                self.beta / self.K, self.rhot, self.taut)
            self.Etinvinv = 1. / self.Etinv
        if self.sf_prior:
            self.Ea, self.Eloga = utils.compute_gamma_expectation(self.nua,
                                                                  self.rhoa)

    def update(self, disp=0):
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
        xtwid = self._xtwid(goodk)
        c = np.mean(self.X / xtwid)
        print('Optimal scale for updating W: {}'.format(c))
        xxtwidinvsq = self.X / c * xtwid**(-2)
        xbarinv = 1. / self._xbar(goodk)
        dEt = self.Et[goodk]
        dEtinvinv = self.Etinvinv[goodk]

        self.tauw[:, goodk] = self.Ewinvinv[:, goodk]**2 * \
                np.dot(xxtwidinvsq, dEtinvinv * self.Ehinvinv[goodk, :].T)
        self.tauw[self.tauw < 1e-100] = 0

        if self.sf_prior:
            self.rhow[:, goodk] = self.gamma * np.exp(np.sum(
                self.logEexpa[:, :, goodk], axis=1))
            self.rhow[:, goodk] = self.rhow[:, goodk] + np.dot(
                xbarinv, dEt * self.Eh[goodk, :].T)
            self.Ew[:, goodk], self.Ewinv[:, goodk] = \
                    utils.compute_gig_expectations(
                        self.gamma,
                        self.rhow[:, goodk],
                        self.tauw[:, goodk])
        else:
            self.rhow[:, goodk] = self.a + np.dot(xbarinv, dEt *
                                                  self.Eh[goodk, :].T)
            self.Ew[:, goodk], self.Ewinv[:, goodk] = \
                    utils.compute_gig_expectations(
                        self.a,
                        self.rhow[:, goodk],
                        self.tauw[:, goodk])
        self.Ewinvinv[:, goodk] = 1. / self.Ewinv[:, goodk]

    def update_h(self):
        goodk = self.goodk()
        xtwid = self._xtwid(goodk)
        c = np.mean(self.X / xtwid)
        print('Optimal scale for updating H: {}'.format(c))
        xxtwidinvsq = self.X / c * xtwid**(-2)
        xbarinv = 1. / self._xbar(goodk)
        dEt = self.Et[goodk]
        dEtinvinv = self.Etinvinv[goodk]
        self.rhoh[goodk, :] = self.b + np.dot(dEt[:, np.newaxis] *
                                              self.Ew[:, goodk].T,
                                              xbarinv)
        self.tauh[goodk, :] = self.Ehinvinv[goodk, :]**2 * \
                np.dot(dEtinvinv[:, np.newaxis] * self.Ewinvinv[:, goodk].T,
                        xxtwidinvsq)
        self.tauh[self.tauh < 1e-100] = 0
        self.Eh[goodk, :], self.Ehinv[goodk, :] = \
                utils.compute_gig_expectations(
                    self.b,
                    self.rhoh[goodk, :],
                    self.tauh[goodk, :])
        self.Ehinvinv[goodk, :] = 1. / self.Ehinv[goodk, :]

    def update_theta(self):
        goodk = self.goodk()
        xtwid = self._xbar(goodk)
        c = np.mean(self.X / xtwid)
        print('Optimal scale for updating theta: {}'.format(c))
        xxtwidinvsq = self.X / c * xtwid**(-2)
        xbarinv = 1. / self._xbar(goodk)
        self.rhot[goodk] = self.beta + np.sum(np.dot(
            self.Ew[:, goodk].T, xbarinv) *
            self.Eh[goodk, :], axis=1)
        self.taut[goodk] = self.Etinvinv[goodk]**2 * \
                np.sum(np.dot(self.Ewinvinv[:, goodk].T, xxtwidinvsq) *
                       self.Ehinvinv[goodk, :], axis=1)
        self.taut[self.taut < 1e-100] = 0
        self.Et[goodk], self.Etinv[goodk] = utils.compute_gig_expectations(
            self.beta / self.K, self.rhot[goodk], self.taut[goodk])
        self.Etinvinv[goodk] = 1. / self.Etinv[goodk]

    def goodk(self):
        if not self.GaP:
            return np.arange(self.K)
        pass

    def clear_badk(self):
        pass

    def bound(self):
        score = 0
        goodk = self.goodk()
        c = np.mean(self.X / self._xtwid(goodk))
        xbar = self._xbar(goodk)

        score = score - np.sum(np.log(xbar) + log(c))
        if self.sf_prior:
            score = score + utils.gig_gamma_term(self.Ew, self.Ewinv, self.rhow,
                                                self.tauw, self.gamma, self.gamma *
                                                np.exp(np.sum(self.logEexpa,
                                                            axis=1)))
            score = score + utils.gamma_term(self.Ea, self.Eloga,
                                             self.nua, self.rhoa,
                                             self.alpha[:, np.newaxis],
                                             self.alpha[:, np.newaxis])
        else:
            score = score + utils.gig_gamma_term(self.Ew, self.Ewinv, self.rhow, self.tauw,
                                     self.a, self.a)

        score = score + utils.gig_gamma_term(self.Eh, self.Ehinv, self.rhoh,
                                            self.tauh, self.b, self.b)
        if self.GaP:
            score = score + utils.gig_gamma_term(self.Et, self.Etinv, self.rhot,
                                                self.taut, self.beta / self.K,
                                                self.beta)
        return score

    def figures(self):
        ''' Animation-type of figures can only be created with PyGTK backend
        '''
        plt.subplot(3, 2, 1)
        specshow(np.log(self.Ew))
        plt.title('E[W]')
        plt.xlabel('component index')
        plt.ylabel('frequency')

        plt.subplot(3, 2, 2)
        specshow(np.log(self.Eh))
        plt.title('E[H]')
        plt.xlabel('time')
        plt.ylabel('component index')

        plt.subplot(3, 2, 3)
        plt.bar(np.arange(self.K), self.Et)
        plt.title('E[theta]')
        plt.xlabel('component index')
        plt.ylabel('E[theta]')

        if self.sf_prior:
            plt.subplot(3, 2, 4)
            specshow(self.Ea)
            plt.title('E[A]')
            plt.xlabel('component index')
            plt.ylabel('filters index')

        plt.subplot(3, 2, 5)
        specshow(np.log(self.X))
        plt.title('Original Spectrogram')
        plt.xlabel('time')
        plt.ylabel('frequency')

        plt.subplot(3, 2, 6)
        specshow(np.log(self._xbar()))
        plt.title('Reconstructed Spectrogram')
        plt.xlabel('time')
        plt.ylabel('frequency')

        time.sleep(0.000001)

    def _xbar(self, goodk=None):
        if goodk is None:
            goodk = np.arange(self.K)
        return np.dot(self.Ew[:, goodk] * self.Et[goodk],
                      self.Eh[goodk, :])

    def _xexplog(self, goodk):
        '''
        sum_k exp(E[log theta_k * W_k * H_k])
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
