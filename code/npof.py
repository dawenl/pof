"""

Non-negative Product-of-Filter model

CREATED: 2013-07-12 11:09:44 by Dawen Liang <daliang@adobe.com>

"""

import time
import numpy as np

from scipy import optimize, special, weave
from joblib import Parallel, delayed

import _pof


class ProductOfFilterLearning:
    def __init__(self, X, n_filters=None, U=None, gamma=None, alpha=None,
                 max_steps=100, n_jobs=1, tol=0.0005, smoothness=100,
                 cold_start=False, random_state=None, verbose=False):
        self.X = X.copy()
        self.n, self.m = X.shape
        self.n_filters = n_filters
        self.max_steps = max_steps
        self.n_jobs = n_jobs
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.smoothness = smoothness
        self.cold_start = cold_start

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        if U is not None:
            self.n_filters = U.shape[0]
            self.U = U.copy()
            self.gamma = gamma.copy()
            self.alpha = alpha.copy()
        else:
            self._init_params()
        self._init_variational()

    def _init_params(self):
        # model parameters
        self.U = np.random.randn(self.n_filters, self.m)
        self.alpha = np.random.gamma(self.smoothness,
                                     1. / self.smoothness,
                                     size=(self.n_filters,))
        self.gamma = np.random.gamma(self.smoothness,
                                     1. / self.smoothness,
                                     size=(self.m,))

    def _init_variational(self):
        self.nu = self.smoothness * np.random.gamma(self.smoothness,
                                                    1. / self.smoothness,
                                                    size=(self.n,
                                                          self.n_filters))
        self.rho = self.smoothness * np.random.gamma(self.smoothness,
                                                     1. / self.smoothness,
                                                     size=(self.n,
                                                           self.n_filters))
        self.EA, self.ElogA = comp_expect(self.nu, self.rho)

    def fit(self):
        old_obj = -np.inf
        for i in xrange(self.max_steps):
            self.transform()
            self._update_params()
            score = self._bound()
            improvement = (score - old_obj) / abs(old_obj)
            if self.verbose:
                print('After ITERATION: %d\tObjective: %.2f\t'
                      'Old objective: %.2f\t'
                      'Improvement: %.5f' % (i, score, old_obj,
                                             improvement))
            if improvement < self.tol:
                break
            old_obj = score
        return self

    def transform(self):
        if self.cold_start:
            # re-initialize all the variational parameters
            self._init_variational()

        if self.verbose:
            last_score = self._bound()
            print('Update (initial)\tObj: {:.2f}'.format(last_score))
            start_t = time.time()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(global_transform)(
                self.X[i], self.n_filters,
                self.U, self.gamma, self.alpha,
                self.nu[i], self.rho[i],
                self.verbose
            )
            for i in xrange(self.n)
        )
        nu_and_rho = np.array(results)
        self.nu, self.rho = nu_and_rho[:, 0, :].copy(), nu_and_rho[:, 1, :].copy()
        self.EA, self.ElogA = comp_expect(self.nu, self.rho)

        if self.verbose:
            t = time.time() - start_t
            score = self._bound()
            print_increment('A', last_score, score)
            print 'Batch update A\ttime: {:.2f}'.format(t)

    def _update_params(self):
        if self.verbose:
            last_score = self._bound()
            start_t = time.time()

        U = Parallel(n_jobs=self.n_jobs)(
            delayed(global_update_U)(
                self.X[:, j], self.U[:, j], self.gamma[j], self.alpha,
                self.nu, self.rho, self.EA, self.ElogA
            )
            for j in xrange(self.m)
        )
        U = np.vstack(U).T
        self.U = U.copy()
        if self.verbose:
            score = self._bound()
            print_increment('U', last_score, score)
            last_score = score

        self._update_gamma()
        if self.verbose:
            score = self._bound()
            print_increment('gamma', last_score, score)
            last_score = score

        self._update_alpha()
        if self.verbose:
            score = self._bound()
            print_increment('alpha', last_score, score)

        if self.verbose:
            t = time.time() - start_t
            print('Update free parameters\ttime: %.2f' % t)

    def _update_gamma(self):
        def f(eta):
            gamma = np.exp(eta)
            return -(self.n * np.sum(gamma * eta - special.gammaln(gamma)) +
                     np.sum(gamma * np.log(self.X) - gamma *
                            np.dot(self.EA, self.U) - gamma * self.X * Eexp))

        def df(eta):
            gamma = np.exp(eta)
            return -gamma * (self.n * (eta + 1 - special.psi(gamma)) +
                             np.sum(-np.dot(self.EA, self.U) +
                                    np.log(self.X) - self.X * Eexp, axis=0))

        Eexp = np.exp(comp_logEexp(self.nu, self.rho, self.U))

        eta0 = np.log(self.gamma)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.gamma = np.exp(eta_hat)
        if self.verbose and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(f(eta_hat), d['task'])
            else:
                print 'f={}, {}'.format(f(eta_hat), d['warnflag'])
            app_grad = approx_grad(f, eta_hat)
            ana_grad = df(eta_hat)
            for idx in xrange(self.m):
                print_gradient('Gamma[{:3d}]'.format(idx), self.gamma[idx],
                               ana_grad[idx], app_grad[idx])

    def _update_alpha(self):
        def f(eta):
            tmp1 = np.exp(eta) * eta - special.gammaln(np.exp(eta))
            tmp2 = self.ElogA * (np.exp(eta) - 1) - self.EA * np.exp(eta)
            return -(self.n * tmp1.sum() + tmp2.sum())

        def df(eta):
            return -np.exp(eta) * (self.n * (eta + 1 -
                                             special.psi(np.exp(eta)))
                                   + np.sum(self.ElogA - self.EA, axis=0))

        eta0 = np.log(self.alpha)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.alpha = np.exp(eta_hat)
        if self.verbose and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(f(eta_hat), d['task'])
            else:
                print 'f={}, {}'.format(f(eta_hat), d['warnflag'])
            app_grad = approx_grad(f, eta_hat)
            ana_grad = df(eta_hat)
            for l in xrange(self.n_filters):
                print_gradient('Alpha[{:3d}]'.format(l), self.alpha[l],
                               ana_grad[l], app_grad[l])

    def _bound(self):
        Eexp = np.exp(comp_logEexp(self.nu, self.rho, self.U))
        # E[log P(w|a)]
        bound = self.n * np.sum(self.gamma * np.log(self.gamma) -
                                special.gammaln(self.gamma))
        bound = bound + np.sum(-self.gamma * np.dot(self.EA, self.U) +
                               (self.gamma - 1) * np.log(self.X) -
                               self.X * Eexp * self.gamma)
        # E[log P(a)]
        bound = bound + self.n * np.sum(self.alpha * np.log(self.alpha) -
                                        special.gammaln(self.alpha))
        bound = bound + np.sum(self.ElogA * (self.alpha - 1) -
                               self.EA * self.alpha)
        # E[loq q(a)]
        bound = bound + np.sum(entropy(self.nu, self.rho))
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
                        printf("U=%.2f\tb=%.2f", U[(l*F) + f], b[(t*L) + l]);
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


def global_transform(x, n_filters,
                     U, gamma, alpha,
                     nu_init, rho_init,
                     verbose):
    nu, rho = _pof.encoder(x, n_filters,
                           U, gamma, alpha,
                           nu_init, rho_init,
                           verbose)
    return (nu, rho)


def global_update_U(x,
                    u_init, gamma, alpha,
                    nu, rho,
                    EA, ElogA):
    u = _pof.update_u(x,
                      u_init, gamma, alpha,
                      nu, rho,
                      EA, ElogA)
    return u
