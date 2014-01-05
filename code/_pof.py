#!/usr/bin/env python

import numpy as np
from scipy import optimize, special, weave


def encoder(x, n_filters, U, gamma, alpha, nu0, rho0, verbose):
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

    theta0 = np.hstack((np.log(nu0), np.log(rho0)))
    theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
    if verbose and d['warnflag']:
        if d['warnflag'] == 2:
            print 'A[]: {}, f={}'.format(d['task'], f(theta_hat))
        else:
            print 'A[, :]: {}, f={}'.format(d['warnflag'], f(theta_hat))
        app_grad = approx_grad(f, theta_hat)
        ana_grad = df(theta_hat)
        for l in xrange(n_filters):
            print_gradient('log_a[, %:3d]' % l, theta_hat[l], ana_grad[l],
                           app_grad[l])
            print_gradient('log_b[, %:3d]' % l, theta_hat[l + n_filters],
                           ana_grad[l + n_filters], app_grad[l + n_filters])

    nu, rho = np.exp(theta_hat[:n_filters]), np.exp(theta_hat[-n_filters:])
    return (nu, rho)


def update_u(x, u0, gamma, alpha, nu, rho, EA, ElogA):
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

    u, _, _ = optimize.fmin_l_bfgs_b(fun, u0, fprime=dfun, disp=0)
    return u


def update_gamma(X, n, U, gamma0, nu, rho, EA, disp=1):
    def f(eta):
        gamma = np.exp(eta)
        return -(n * np.sum(gamma * eta - special.gammaln(gamma)) +
                 np.sum(gamma * np.log(X) - gamma * EA.dot(U) -
                        gamma * X * Eexp))

    def df(eta):
        gamma = np.exp(eta)
        return -gamma * (n * (eta + 1 - special.psi(gamma)) +
                         np.sum(-EA.dot(U) + np.log(X) - X * Eexp, axis=0))

    Eexp = np.exp(comp_logEexp(nu, rho, U))

    eta0 = np.log(gamma0)
    eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
    gamma = np.exp(eta_hat)

    if disp and d['warnflag']:
        if d['warnflag'] == 2:
            print 'f={}, {}'.format(f(eta_hat), d['task'])
        else:
            print 'f={}, {}'.format(f(eta_hat), d['warnflag'])
        app_grad = approx_grad(f, eta_hat)
        ana_grad = df(eta_hat)
        for idx in xrange(X.shape[1]):
            print_gradient('Gamma[{:3d}]'.format(idx), gamma[idx],
                           ana_grad[idx], app_grad[idx])
    return gamma


def update_alpha(X, n, alpha0, EA, ElogA):
    def f(eta):
        tmp1 = np.exp(eta) * eta - special.gammaln(np.exp(eta))
        tmp2 = ElogA * (np.exp(eta) - 1) - EA * np.exp(eta)
        return -(n * tmp1.sum() + tmp2.sum())

    def df(eta):
        return -np.exp(eta) * (n * (eta + 1 - special.psi(np.exp(eta))) +
                               np.sum(ElogA - EA, axis=0))

    eta0 = np.log(alpha0)
    eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
    alpha = np.exp(eta_hat)
    return alpha


def _init_variational_params(n_filters, n_sampels, smoothness):
    nu = smoothness * np.random.gamma(smoothness,
                                      1. / smoothness,
                                      size=(n_sampels, n_filters))
    rho = smoothness * np.random.gamma(smoothness,
                                       1. / smoothness,
                                       size=(n_sampels, n_filters))
    return (nu, rho)


def _init_params(n_filters, n_feats, smoothness):
    U = np.random.randn(n_filters, n_feats)
    gamma = np.random.gamma(smoothness, 1. / smoothness, size=(n_feats,))
    alpha = np.random.gamma(smoothness, 1. / smoothness, size=(n_filters,))
    return (U, gamma, alpha)


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


def comp_expect(nu, rho):
    return (nu / rho, special.psi(nu) - np.log(rho))


def entropy(nu, rho):
    return (nu - np.log(rho) + special.gammaln(nu) +
            (1 - nu) * special.psi(nu))


def comp_logEexp(a, b, U):
    """ Compute log[E(\prod_l exp(U_{fl} A_{lt}))] where
    A_{lt} ~ Gamma(a_{lt}, b_{lt})
    """
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


def print_gradient(name, val, grad, approx):
    print('{} = {:.2f}\tGradient: {:.2f}\tApprox: {:.2f}\t'
          '| Diff |: {:.3f}'.format(name, val, grad, approx,
                                    np.abs(grad - approx)))


def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2 * delta)
    return grad


def print_increment(name, last_score, score):
    diff_str = '+' if score > last_score else '-'
    print('Update ({})\tBefore: {:.2f}\tAfter: {:.2f}\t{}'.format(
        name, last_score, score, diff_str))

