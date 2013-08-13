"""

Source-filter dictionary prior GIG-NMF

CREATED: 2013-08-08 13:43:27 by Dawen Liang <daliang@adobe.com>

"""

import functools
from math import log
import time

import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as optimize
import scipy.special as special

import _gap


specshow = functools.partial(plt.imshow, cmap=plt.cm.jet, origin='lower',
                             aspect='auto', interpolation='nearest')


class SF_GIG_NMF:
    def __init__(self, X, U, gamma, alpha, K=100, smoothness=100,
                 seed=None, **kwargs):
        self.X = X.copy()
        self.K = K
        self.U = U.copy()
        self.alpha = alpha.copy()
        self.gamma = gamma.copy()
        # check if gamma is in the shape of (F, 1) for broadcasting with (F, K)
        if self.gamma.ndim == 1:
            self.gamma = self.gamma[:, np.newaxis]

        self.F, self.T = X.shape
        self.L = alpha.size

        if seed is None:
            print 'Using random seed'
        else:
            print 'Using fixed seed {}'.format(seed)

        self._parse_args(**kwargs)
        self._init(smoothness)

    def _parse_args(self, **kwargs):
        self.b = float(kwargs['b']) if 'b' in kwargs else 0.1

    def _init(self, smoothness):
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
        self.nua = 10000 * np.random.gamma(smoothness,
                                           1. / smoothness,
                                           size=(self.L, self.K))
        self.rhoa = 10000 * np.random.gamma(smoothness,
                                            1. / smoothness,
                                            size=(self.L, self.K))
        self.compute_expectations()
        self.logEexpa = np.zeros((self.F, self.L, self.K))

    def compute_expectations(self):
        self.Ew, self.Ewinv = _gap.compute_gig_expectations(self.gamma,
                                                            self.rhow,
                                                            self.tauw)
        self.Ewinvinv = 1. / self.Ewinv
        self.Eh, self.Ehinv = _gap.compute_gig_expectations(self.b,
                                                            self.rhoh,
                                                            self.tauh)
        self.Ehinvinv = 1. / self.Ehinv
        self.Ea, self.Eloga = _gap.compute_gamma_expectation(self.nua,
                                                             self.rhoa)

    def update(self, disp=0):
        ''' Do optimization for one iteration
        '''
        self.update_h()
        self.update_w()
        for k in xrange(self.K):
            self.update_a(k, disp)

    def update_a(self, k, disp):
        def f(theta):
            nu, rho = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Eloga = _gap.compute_gamma_expectation(nu, rho)
            # E[p(W|a)] + E[p(a)] - E[q(a)], collect terms by sufficient
            # statistics
            val = np.sum((self.alpha - nu) * Eloga)
            val = val - np.sum((self.alpha - rho) * Ea)
            val = val + np.sum(special.gammaln(nu) - nu * np.log(rho))

            logEexp = _gap.comp_log_exp(nu, rho, self.U)
            val = val - np.sum(self.gamma.ravel() * (self.Ew[:, k] *
                               np.exp(np.sum(logEexp, axis=1)) +
                               np.dot(self.U, Ea)))
            return -val

        def df(theta):
            nu, rho = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            logEexp = _gap.comp_log_exp(nu, rho, self.U)

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
        self.Ea[:, k], self.Eloga[:, k] = _gap.compute_gamma_expectation(
            self.nua[:, k], self.rhoa[:, k])
        self.logEexpa[:, :, k] = _gap.comp_log_exp(self.nua[:, k],
                                                   self.rhoa[:, k],
                                                   self.U)

    def update_w(self):
        xtwid = self._xtwid()
        c = np.mean(self.X / xtwid)
        print('Optimal scale for updating W: {}'.format(c))
        xxtwidinvsq = self.X / c * xtwid**(-2)
        xbarinv = 1. / self._xbar()

        self.rhow = self.gamma * np.exp(np.sum(self.logEexpa, axis=1))
        self.rhow = self.rhow + np.dot(xbarinv, self.Eh.T)
        self.tauw = self.Ewinvinv**2 * np.dot(xxtwidinvsq, self.Ehinvinv.T)
        self.tauw[self.tauw < 1e-100] = 0

        self.Ew, self.Ewinv = _gap.compute_gig_expectations(self.gamma,
                                                            self.rhow,
                                                            self.tauw)
        self.Ewinvinv = 1. / self.Ewinv

    def update_h(self):
        xtwid = self._xtwid()
        c = np.mean(self.X / xtwid)
        print('Optimal scale for updating H: {}'.format(c))
        xxtwidinvsq = self.X / c * xtwid**(-2)
        xbarinv = 1. / self._xbar()
        self.rhoh = self.b + np.dot(self.Ew.T, xbarinv)
        self.tauh = self.Ehinvinv**2 * np.dot(self.Ewinvinv.T, xxtwidinvsq)
        self.tauh[self.tauh < 1e-100] = 0
        self.Eh, self.Ehinv = _gap.compute_gig_expectations(self.b,
                                                            self.rhoh,
                                                            self.tauh)
        self.Ehinvinv = 1. / self.Ehinv

    def bound(self):
        score = 0
        c = np.mean(self.X / self._xtwid())
        xbar = self._xbar()

        score = score - np.sum(np.log(xbar) + log(c))
        score = score + _gap.gig_gamma_term(self.Ew, self.Ewinv, self.rhow,
                                            self.tauw, self.gamma, self.gamma *
                                            np.exp(np.sum(self.logEexpa,
                                                          axis=1)))
        score = score + _gap.gig_gamma_term(self.Eh, self.Ehinv, self.rhoh,
                                            self.tauh, self.b, self.b)
        score = score + _gap.gamma_term(self.Ea, self.Eloga, self.nua,
                                        self.rhoa, self.alpha)
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

    def _xbar(self):
        return np.dot(self.Ew, self.Eh)

    def _xtwid(self):
        return np.dot(self.Ewinvinv, self.Ehinvinv)
