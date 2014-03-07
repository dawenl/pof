"""

Non-negative Product-of-Filter model

CREATED: 2013-07-12 11:09:44 by Dawen Liang <dliang@ee.columbia.edu>

"""

import time
import numpy as np

from scipy import io, optimize, special, weave
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, TransformerMixin


class ProductOfFiltersLearning(BaseEstimator, TransformerMixin):
    '''Product-of-Filters learning'''

    def __init__(self, n_feats=None, n_filters=None, U=None, gamma=None,
                 alpha=None, max_steps=50, n_jobs=1, tol=0.0005,
                 save_filters=False, smoothness=100, random_state=None,
                 verbose=False):
        '''Product-of-Filters learning

        Arguments
        ---------
        n_feats : int
            The dimension of the data to be modeled

        n_filters : int
            Number of filters to to extract

        U, gamma, alpha : numpy-array
            Filter parameters

        max_steps : int
            Maximal number of iterations to perform

        n_jobs : int
            Number of parallel jobs to run

        tol : float
            The threshold on the increase of the objective to stop the
            iteration

        save_filters : bool
            Save the intermediate filter parameters after each iteration or not

        smoothness : int
            Smoothness on the initialization variational parameters

        random_state : int or RandomState
            Pseudo random number generator used for sampling

        verbose : bool
            Whether to show progress during training
        '''
        self.n_feats = n_feats
        self.n_filters = n_filters
        self.max_steps = max_steps
        self.n_jobs = n_jobs
        self.tol = tol
        self.save_filters = save_filters
        self.random_state = random_state
        self.verbose = verbose
        self.smoothness = smoothness

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        if U is not None:
            self.set_filters(U, gamma, alpha)
        else:
            self._init_filters()

    def _init_filters(self):
        # model parameters
        self.U = np.random.randn(self.n_filters, self.n_feats)
        self.alpha = np.random.gamma(self.smoothness,
                                     1. / self.smoothness,
                                     size=(self.n_filters,))
        self.gamma = np.random.gamma(self.smoothness,
                                     1. / self.smoothness,
                                     size=(self.n_feats,))

    def _init_variational(self, n_samples):
        self.nu = self.smoothness * np.random.gamma(self.smoothness,
                                                    1. / self.smoothness,
                                                    size=(n_samples,
                                                          self.n_filters))
        self.rho = self.smoothness * np.random.gamma(self.smoothness,
                                                     1. / self.smoothness,
                                                     size=(n_samples,
                                                           self.n_filters))
        self.EA, self.ElogA = comp_expect(self.nu, self.rho)

    def _save_filters(self, fname, save_EA=False):
        out_data = {'U': self.U,
                    'gamma': self.gamma,
                    'alpha': self.alpha}
        if save_EA:
            out_data['EA'] = self.EA
        io.savemat(fname, out_data)

    def set_filters(self, U, gamma, alpha):
        '''Set the filter parameters.

        Parameters
        ----------
        U : numpy-array, shape (n_filters, n_feats)
            Filters

        gamma : numpy-array, shape (n_feats, )
            Frequency-dependent noise level

        alpha : numpy-array, shape (n_filters, )
            Filter-specific sparsity

        Returns
        -------
        self : object
            Return the instance itself.
        '''
        self.n_filters, self.n_feats = U.shape
        #self.U, self.gamma, self.alpha = U, gamma, alpha
        self.U = U.copy()
        self.gamma = gamma.copy()
        self.alpha = alpha.copy()
        return self

    def fit(self, X):
        '''Fit the model to the data in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)
            Training data.

        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        if X.shape[1] != self.n_feats:
            raise ValueError('The number of dimension of data does not match.')

        old_obj = -np.inf
        for i in xrange(self.max_steps):
            self.transform(X)
            self._update_filters(X)
            if self.save_filters:
                self._save_filters('sf_inter_L%d.iter%d.mat' % (self.n_filters,
                                                                i))
            score = self._bound(X)
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

    def transform(self, X, attr=None):
        '''Encode the data as a sparse linear combination of the filters in the
        log-spectral domain.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_feats)

        attr: string
            The name of attribute, default 'EA'. Can be changed to ElogA to
            obtain E_q[log A] as transformed data.

        Returns
        -------
        X_new : array-like, shape(n_samples, n_filters)
            Transformed data, as specified by attr.
        '''
        if attr is None:
            attr = 'EA'

        n_samples = X.shape[0]
        if not hasattr(self, 'nu'):     # or equivalently, rho
            self._init_variational(n_samples)

        if self.verbose:
            last_score = self._bound(X)
            print('Update (initial)\tObj: %.2f' % last_score)
            start_t = time.time()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(global_transform)(
                X[i], self.n_filters,
                self.U, self.gamma, self.alpha,
                self.nu[i], self.rho[i],
                self.verbose
            )
            for i in xrange(n_samples)
        )
        nu_and_rho = np.array(results)
        self.nu = nu_and_rho[:, 0, :].copy()
        self.rho = nu_and_rho[:, 1, :].copy()
        self.EA, self.ElogA = comp_expect(self.nu, self.rho)

        if self.verbose:
            t = time.time() - start_t
            score = self._bound(X)
            print_increment('A', last_score, score)
            print 'Batch update A\ttime: %.2f' % t

        return getattr(self, attr)

    def _update_filters(self, X):
        if self.verbose:
            last_score = self._bound(X)
            start_t = time.time()

        U = Parallel(n_jobs=self.n_jobs)(
            delayed(global_update_U)(
                X[:, j], self.U[:, j], self.gamma[j], self.alpha,
                self.nu, self.rho, self.EA, self.ElogA
            )
            for j in xrange(self.n_feats)
        )
        U = np.vstack(U).T
        self.U = U.copy()
        if self.verbose:
            score = self._bound(X)
            print_increment('U', last_score, score)
            last_score = score

        self._update_gamma(X)
        if self.verbose:
            score = self._bound(X)
            print_increment('gamma', last_score, score)
            last_score = score

        self._update_alpha(X)
        if self.verbose:
            score = self._bound(X)
            print_increment('alpha', last_score, score)

        if self.verbose:
            t = time.time() - start_t
            print('Update free parameters\ttime: %.2f' % t)

    def _update_gamma(self, X):
        def f(eta):
            gamma = np.exp(eta)
            return -(n_samples * np.sum(gamma * eta - special.gammaln(gamma)) +
                     np.sum(gamma * np.log(X) - gamma *
                            self.EA.dot(self.U) - gamma * X * Eexp))

        def df(eta):
            gamma = np.exp(eta)
            return -gamma * (n_samples * (eta + 1 - special.psi(gamma)) +
                             np.sum(-self.EA.dot(self.U) +
                                    np.log(X) - X * Eexp, axis=0))

        n_samples = X.shape[0]
        Eexp = np.exp(comp_logEexp(self.nu, self.rho, self.U))

        eta0 = np.log(self.gamma)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.gamma = np.exp(eta_hat)
        if self.verbose and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f=%.3f, %s' % (f(eta_hat), d['task'])
            else:
                print 'f=%.3f, %d' % (f(eta_hat), d['warnflag'])
            app_grad = approx_grad(f, eta_hat)
            ana_grad = df(eta_hat)
            for idx in xrange(self.n_feats):
                print_gradient('Gamma[%3d]' % idx, self.gamma[idx],
                               ana_grad[idx], app_grad[idx])

    def _update_alpha(self, X):
        def f(eta):
            tmp1 = np.exp(eta) * eta - special.gammaln(np.exp(eta))
            tmp2 = self.ElogA * (np.exp(eta) - 1) - self.EA * np.exp(eta)
            return -(n_samples * tmp1.sum() + tmp2.sum())

        def df(eta):
            return -np.exp(eta) * (n_samples *
                                   (eta + 1 - special.psi(np.exp(eta)))
                                   + np.sum(self.ElogA - self.EA, axis=0))

        n_samples = X.shape[0]
        eta0 = np.log(self.alpha)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.alpha = np.exp(eta_hat)
        if self.verbose and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f=%.3f, %s' % (f(eta_hat), d['task'])
            else:
                print 'f=%.3f, %d' % (f(eta_hat), d['warnflag'])
            app_grad = approx_grad(f, eta_hat)
            ana_grad = df(eta_hat)
            for l in xrange(self.n_filters):
                print_gradient('Alpha[%3d]' % l, self.alpha[l],
                               ana_grad[l], app_grad[l])

    def _bound(self, X):
        n_samples = X.shape[0]
        Eexp = np.exp(comp_logEexp(self.nu, self.rho, self.U))
        # E[log P(w|a)]
        bound = n_samples * np.sum(self.gamma * np.log(self.gamma) -
                                   special.gammaln(self.gamma))
        bound = bound + np.sum(-self.gamma * self.EA.dot(self.U) +
                               (self.gamma - 1) * np.log(X) -
                               X * Eexp * self.gamma)
        # E[log P(a)]
        bound = bound + n_samples * np.sum(self.alpha * np.log(self.alpha) -
                                           special.gammaln(self.alpha))
        bound = bound + np.sum(self.ElogA * (self.alpha - 1) -
                               self.EA * self.alpha)
        # E[loq q(a)]
        bound = bound + np.sum(entropy(self.nu, self.rho))
        return bound


def print_gradient(name, val, grad, approx):
    print('%s = %.2f\tGradient: %.2f\tApprox: %.2f\t'
          '| Diff |: %.3f' % (name, val, grad, approx,
                              np.abs(grad - approx)))


def print_increment(name, last_score, score):
    diff_str = '+' if score > last_score else '-'
    print('Update (%s)\tBefore: %.2f\tAfter: %.2f\t%s' % (
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


def global_transform(x, n_filters,
                     U, gamma, alpha,
                     nu_init, rho_init,
                     verbose):
    def f(theta):
        nu, rho = np.exp(theta[:n_filters]), np.exp(theta[-n_filters:])
        Ea, Eloga = comp_expect(nu, rho)
        logEexp = _comp_logEexp(nu, rho, U, update_U=False)
        likeli = (-x * np.exp(np.sum(logEexp, axis=0)) - Ea.dot(U)) * gamma
        prior = (alpha - 1) * Eloga - alpha * Ea
        ent = entropy(nu, rho)
        return -(likeli.sum() + prior.sum() + ent.sum())

    def df(theta):
        nu, rho = np.exp(theta[:n_filters]), np.exp(theta[-n_filters:])
        logEexp = _comp_logEexp(nu, rho, U, update_U=False)

        tmp = U / rho[:, np.newaxis]
        log_term, inv_term = np.empty_like(tmp), np.empty_like(tmp)
        idx = (tmp > -1)
        log_term[idx] = np.log1p(tmp[idx])
        log_term[-idx] = -np.inf
        inv_term[idx], inv_term[-idx] = 1. / (1. + tmp[idx]), np.inf

        grad_nu = np.sum(x * log_term * np.exp(np.sum(logEexp, axis=0)) * gamma
                         - U / rho[:, np.newaxis] * gamma, axis=1)
        grad_nu = grad_nu + (alpha - nu) * special.polygamma(1, nu)
        grad_nu = grad_nu + 1 - alpha / rho
        grad_rho = nu / rho**2 * np.sum(-U * x * inv_term *
                                        np.exp(np.sum(logEexp, axis=0)) *
                                        gamma + U * gamma, axis=1)
        grad_rho = grad_rho + alpha * (nu / rho**2 - 1. / rho)
        return -np.hstack((nu * grad_nu, rho * grad_rho))

    theta0 = np.hstack((np.log(nu_init), np.log(rho_init)))
    theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
    if verbose and d['warnflag']:
        if d['warnflag'] == 2:
            print 'A[]: %s, f=%.3f' % (d['task'], f(theta_hat))
        else:
            print 'A[, :]: %d, f=%.3f' % (d['warnflag'], f(theta_hat))
        app_grad = approx_grad(f, theta_hat)
        ana_grad = df(theta_hat)
        for l in xrange(n_filters):
            print_gradient('log_a[, %.3d]' % l, theta_hat[l], ana_grad[l],
                           app_grad[l])
            print_gradient('log_b[, %.3d]' % l, theta_hat[l + n_filters],
                           ana_grad[l + n_filters], app_grad[l + n_filters])

    nu, rho = np.exp(theta_hat[:n_filters]), np.exp(theta_hat[-n_filters:])
    return (nu, rho)


def global_update_U(x,
                    u_init, gamma, alpha,
                    nu, rho,
                    EA, ElogA):
    def fun(u):
        Eexp = np.exp(np.sum(_comp_logEexp(nu, rho, u), axis=1))
        return np.sum(gamma * (Eexp * x + EA.dot(u)))

    def dfun(u):
        tmp = 1 + u / rho
        inv_term = np.empty_like(tmp)
        idx = (tmp > 0)
        inv_term[idx], inv_term[-idx] = 1. / tmp[idx], np.inf
        Eexp = np.exp(np.sum(_comp_logEexp(nu, rho, u), axis=1))
        return np.sum(EA * (1 - (x * Eexp)[:, np.newaxis] * inv_term), axis=0)

    u, _, _ = optimize.fmin_l_bfgs_b(fun, u_init, fprime=dfun, disp=0)
    return u
