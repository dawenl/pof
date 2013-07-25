"""

Source-filter dictionary prior learning for gamma noise model

CREATED: 2013-07-12 11:09:44 by Dawen Liang <daliang@adobe.com> 

"""

import sys, time

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
        self.alpha = np.random.gamma(smoothness, 1./smoothness, size=(self.L,))
        self.gamma = np.random.gamma(smoothness, 1./smoothness, size=(self.F,))

    def _init_variational(self, smoothness):
        self.a = smoothness * np.random.gamma(smoothness, 1./smoothness, 
                                              size=(self.T, self.L))
        self.b = smoothness * np.random.gamma(smoothness, 1./smoothness, 
                                              size=(self.T, self.L))
        self.EA, self.ElogA = comp_expect(self.a, self.b)

    def vb_e(self, cold_start=True, batch=True, smoothness=100, maxiter=500,
            atol=1e-3, rtol=1e-5, verbose=True, disp=0):
        """ Perform one variational E-step, which may have one sub-iteration or
        multiple sub-iterations if e_converge is set to True, to appxorimate the 
        posterior P(A | -)

        Parameters
        ----------
        cold_start: bool
            Do e-step with fresh start, otherwise just do e-step with 
            previous values as initialization.
        batch: bool
            Do e-step as a whole optimization if true. Otherwise, do multiple
            sub-iterations until convergence.
        smoothness: float
            Smootheness of the variational initialization, larger value will
            lead to more concentrated initialization.
        maxiter: int
            Maximal number of sub-iterations in one e-step.
        atol: float 
            Absolute convergence threshold. 
        rtol: float
            Relative increase convergence threshold.
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
            last_score = self._vb_bound()
            print('Update (initial)\tObj: {:.2f}'.format(last_score))

        if batch:
            start_t = time.time()
            for t in xrange(self.T):
                self.update_theta_batch(t, disp)
                if verbose and not t % 100:
                    sys.stdout.write('.')
            t = time.time() - start_t
            if verbose:
                sys.stdout.write('\n')
                print 'Batch update\ttime: {:.2f}'.format(t)
        else:
            old_bound = -np.inf
            for i in xrange(maxiter):
                old_a = self.a.copy()
                old_mu = self.mu.copy()
                start_t = time.time()
                for l in xrange(self.L):
                    self.update_theta(l, disp)
                    if verbose and not l % 5:
                        sys.stdout.write('.')
                t = time.time() - start_t
                a_diff = np.mean(np.abs(old_a - self.a))
                mu_diff = np.mean(np.abs(old_mu - self.mu))

                bound = self._vb_bound()
                improvement = (bound - old_bound) / np.abs(old_bound)

                if verbose:
                    sys.stdout.write('\n')
                    print('Subiter: {:3d}\ta diff: {:.4f}\tmu diff: {:.4f}\t'
                          'bound: {:.2f}\tbound improvement: {:.5f}\t'
                          'time: {:.2f}'.format(i, a_diff, mu_diff, bound,
                                                improvement, t))
                if improvement < rtol or (a_diff <= atol and mu_diff <= atol):
                    break
                old_bound = bound
        if verbose:
            obj = self._vb_bound()
            diff_str = '+' if obj > last_score else '-' 
            print('Update (A)\tBefore: {:.2f}\tAfter: {:.2f}\t{}'.format(last_score, obj, diff_str))


    def update_theta_batch(self, t, disp):
        def f(theta):
            a, b = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, Eloga = comp_expect(a, b)
            Eexpa = self.comp_exp_expect(a[:, np.newaxis], b[:, np.newaxis], self.U)

            likeli = (-self.W[t,:] * np.prod(Eexpa, axis=0) - np.dot(Ea, self.U)) * self.gamma
            prior = (self.alpha - 1) * Eloga - self.alpha * Ea
            ent = entropy(a, b) 

            return -(likeli.sum() + prior.sum() + ent.sum())

        def df(theta):
            a, b = np.exp(theta[:self.L]), np.exp(theta[-self.L:])
            Ea, _ = comp_expect(a, b)
            Eexpa = self.comp_exp_expect(a[:, np.newaxis], b[:, np.newaxis], self.U)

            tmp = 1 + self.U/b[:, np.newaxis]
            log_term, inv_term = np.empty_like(tmp), np.empty_like(tmp)
            idx_pos = (tmp > 0)
            idx_nz = (tmp != 0)
            log_term[idx_pos], log_term[-idx_pos] = np.log(tmp[idx_pos]), -np.inf
            inv_term[idx_nz], inv_term[-idx_nz] = 1./tmp[idx_nz], np.inf

            grad_a = a * (np.sum(self.W[t,:] * log_term * np.prod(Eexpa, axis=0) * self.gamma - self.U/b[:, np.newaxis] * self.gamma , axis=1) + (self.alpha - a) * special.polygamma(1, a) + 1 - self.alpha / b)
            grad_b = b * (a/b**2 * np.sum(-self.U * self.W[t,:] * inv_term * np.prod(Eexpa, axis=0) * self.gamma + self.U * self.gamma, axis=1) + self.alpha * (a/b**2 - 1./b))
            return -np.hstack((grad_a, grad_b))

        theta0 = np.hstack((np.log(self.a[t,:]), np.log(self.b[t,:])))
        theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'A[{}, :]: {}, f={}'.format(t, d['task'], f(theta_hat))
            else:
                print 'A[{}, :]: {}, f={}'.format(t, d['warnflag'], f(theta_hat))
            app_grad = approx_grad(f, theta_hat)
            for l in xrange(self.L):
                print_gradient('log_a[{}, {:3d}]'.format(t, l), theta_hat[l],
                        df(theta_hat)[l], app_grad[l])
                print_gradient('log_b[{}, {:3d}]'.format(t, l), theta_hat[l +
                    self.L], df(theta_hat)[l + self.L], app_grad[l + self.L])

        self.a[t,:], self.b[t,:] = np.exp(theta_hat[:self.L]), np.exp(theta_hat[-self.L:])
        assert(np.all(self.a[t,:] > 0))
        assert(np.all(self.b[t,:] > 0))
        self.EA[t,:], self.ElogA[t,:] = comp_expect(self.a[t,:], self.b[t,:])

    def update_theta(self, l, disp):                
        #def f(theta):
        #    a, mu = np.exp(theta[:self.T]), np.exp(theta[-self.T:])
        #    Ea, Ea2, Eloga = comp_expect(a, mu)

        #    const = (self.alpha[l] - 1) * Eloga + entropy(a, mu) 
        #    return -np.sum(const + Ea * lcoef + Ea2 * qcoef)
        #        
        #def df(theta):
        #    a, mu = np.exp(theta[:self.T]), np.exp(theta[-self.T:])

        #    grad_a = a * (-mu**2/a**2 * qcoef + (self.alpha[l] - a) * special.polygamma(1, a) - self.alpha[l] / a + 1)
        #    grad_mu = mu * (lcoef + 2 * (mu + mu/a) * qcoef + self.alpha[l]/mu) 
        #    return -np.hstack((grad_a, grad_mu))

        #Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        #lcoef = np.sum(Eres * self.U[l, :] * self.gamma, axis=1) - self.alpha[l]
        #qcoef = -np.sum(self.gamma * self.U[l,:]**2)/2

        #theta0 = np.hstack((np.log(self.a[:,l]), np.log(self.mu[:,l])))

        #theta_hat, _, d = optimize.fmin_l_bfgs_b(f, theta0, fprime=df, disp=0)
        #if disp and d['warnflag']:
        #    if d['warnflag'] == 2:
        #        print 'A[:, {}]: {}, f={}'.format(l, d['task'], f(theta_hat))
        #    else:
        #        print 'A[:, {}]: {}, f={}'.format(l, d['warnflag'], f(theta_hat))
        #    app_grad = approx_grad(f, theta_hat)
        #    for t in xrange(self.T):
        #        print 'a[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[t], app_grad[t], df(theta_hat)[t], np.abs(app_grad[t] - df(theta_hat)[t]))
        #    for t in xrange(self.T):
        #        print 'mu[{:3d}, {}] = {:.3f}\tApproximated: {:.5f}\tGradient: {:.5f}\t|Approximated - True|: {:.5f}'.format(t, l, theta_hat[t + self.T], app_grad[t + self.T], df(theta_hat)[t + self.T], np.abs(app_grad[t + self.T] - df(theta_hat)[t + self.T]))

        #self.a[:,l], self.mu[:,l] = np.exp(theta_hat[:self.T]), np.exp(theta_hat[-self.T:])

        #assert(np.all(self.a[:,l] > 0))
        #assert(np.all(self.mu[:,l] > 0))
        #self.EA[:,l], self.EA2[:,l], self.ElogA[:,l] = comp_expect(self.a[:,l], self.mu[:,l])
        raise NotImplementedError()

    def vb_m(self, batch=False, atol=1e-3, verbose=True, disp=0, update_alpha=True):
        """ Perform one M-step, update the model parameters with A fixed from E-step

        Parameters
        ----------
        batch: bool
            Update U as a whole optimization if true. Otherwise, update U across
            different basis.
        atol: float
            Absolute convergence threshold.
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
        start_t = time.time()
        if verbose:
            last_score = self.bound()
            print('Update (initial)\tObj: {:.2f}'.format(last_score))
        if batch:
            self.update_u_batch(disp) 
            if verbose:
                obj = self.bound()
                diff_str = '+' if obj > last_score else '-'
                print('Update (U)\tBefore: {:.2f}\tAfter: {:.2f}\t{}'.format(last_score,
                    obj, diff_str))
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
        
        t = time.time() - start_t
        U_diff = np.mean(np.abs(self.U - old_U))
        sigma_diff = np.mean(np.abs(np.sqrt(1./self.gamma) - np.sqrt(1./old_gamma)))
        alpha_diff = np.mean(np.abs(self.alpha - old_alpha))
        if verbose:
            print('U diff: {:.4f}\tsigma dff: {:.4f}\talpha diff: {:.4f}\t'
                    'time: {:.2f}'.format(U_diff, sigma_diff, alpha_diff, t))
        if U_diff < atol and sigma_diff < atol and alpha_diff < atol:
            return True
        return False

    def update_u_batch(self, disp):
        def f_df(u):
            U = u.reshape(self.L, self.F) 
            Eexp = 1.
            for l in xrange(self.L):
                Eexp *= self.comp_exp_expect(self.a[:, l, np.newaxis], self.b[:, l,
                    np.newaxis], U[l, :])
            grad_U = np.zeros_like(U)
            for l in xrange(self.L):
                tmp = 1 + U[l, :]/self.b[:, l, np.newaxis]
                inv_term = np.empty_like(tmp)
                idx = (tmp > 0)
                inv_term[idx], inv_term[-idx] = 1./tmp[idx], np.inf
                grad_U[l, :] = np.sum(self.EA[:, l, np.newaxis] * (1 - self.W *
                    Eexp * inv_term))
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
            Eexpa = self.comp_exp_expect(self.a[:, l, np.newaxis], self.b[:, l, np.newaxis], u)
            return np.sum(np.outer(self.EA[:,l], u) + self.W * Eexpa * Eres)
        
        def df(u):
            tmp = self.comp_exp_expect(self.a[:, l, np.newaxis] + 1, self.b[:, l, np.newaxis], u) 
            return np.sum(self.EA[:,l, np.newaxis] * (1 - self.W * Eres * tmp), axis=0) 

        k_idx = np.delete(np.arange(self.L), l)
        Eres = 1.
        for k in k_idx:
            Eres *= self.comp_exp_expect(self.a[:, k, np.newaxis], self.b[:, k,
                np.newaxis], self.U[k, :])
        u0 = self.U[l,:]
        self.U[l,:], _, d = optimize.fmin_l_bfgs_b(f, u0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'U[{}, :]: {}, f={}'.format(l, d['task'], f(self.U[l,:]))
            else:
                print 'U[{}, :]: {}, f={}'.format(l, d['warnflag'], f(self.U[l,:]))
            app_grad = approx_grad(f, self.U[l,:])
            for fr in xrange(self.F):
                print_gradient('U[{}, {:3d}]'.format(l, fr), self.U[l, fr],
                        df(self.U[l,:])[fr], app_grad[fr])

    def update_gamma(self, disp):
        def f(eta):
            gamma = np.exp(eta)
            return -(self.T * np.sum(gamma * eta - special.gammaln(gamma)) +
                    np.sum(gamma * np.log(self.W) - gamma * np.dot(self.EA, 
                        self.U) - gamma * self.W * Eexp))

        def df(eta):
            gamma = np.exp(eta)
            return -gamma * (self.T * (eta + 1 - special.psi(gamma)) + 
                    np.sum(-np.dot(self.EA, self.U) + np.log(self.W) - 
                        self.W * Eexp, axis=0))

        Eexp = 1.
        for l in xrange(self.L):
            Eexp *= self.comp_exp_expect(self.a[:, l, np.newaxis], self.b[:, l,
                np.newaxis], self.U[l, :])

        eta0 = np.log(self.gamma)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.gamma = np.exp(eta_hat)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(f(eta_hat), d['task'])
            else:
                print 'f={}, {}'.format(f(eta_hat), d['warnflag'])
            app_grad = approx_grad(f, eta_hat)
            for idx in xrange(self.F):
                print_gradient('Gamma[{:3d}]'.format(idx), self.gamma[idx],
                        df(eta_hat)[idx], app_grad[idx])

    def update_alpha(self, disp):
        def f(eta):
            tmp1 = np.exp(eta) * eta - special.gammaln(np.exp(eta))
            tmp2 = self.ElogA * (np.exp(eta) - 1) - self.EA * np.exp(eta)
            return -(self.T * tmp1.sum() + tmp2.sum())

        def df(eta):
            return -np.exp(eta) * (self.T * (eta + 1 - special.psi(np.exp(eta)))
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
            for l in xrange(self.L):
                print_gradient('Alpha[{:3d}]'.format(l), self.alpha[l], 
                        df(eta_hat)[l], app_grad[l])

    def _vb_bound(self):
        vbound = self.bound()
        vbound += np.sum(entropy(self.a, self.b))
        return vbound

    def bound(self):
        Eexp = 1.
        for l in xrange(self.L):
            Eexp *= self.comp_exp_expect(self.a[:, l, np.newaxis], self.b[:, l,
                np.newaxis], self.U[l, :])
        # E[log P(w|a)]
        bound = self.T * np.sum(self.gamma * np.log(self.gamma) -
                special.gammaln(self.gamma))
        bound = bound + np.sum(-self.gamma * np.dot(self.EA, self.U) + (self.gamma -
            1) * np.log(self.W) - self.W * Eexp * self.gamma)
        # E[log P(a)]
        bound = bound + self.T * np.sum(self.alpha * np.log(self.alpha) - 
                special.gammaln(self.alpha))
        bound = bound + np.sum(self.ElogA * (self.alpha - 1) - self.EA * self.alpha)
        return bound 

    def comp_exp_expect(self, alpha, beta, U):
        ''' Compute E[exp(-au)] where a ~ Gamma(alpha, beta) and u constant
            
        This function makes extensive use of broadcast, thus the dimension of 
        input arguments can only be one of the following two situations:
             1) U has shape (L, F), alpha and beta have shape (L, 1)
                --> output shape (L, F)
             2) U has shape (F, ), alpha and beta have shape (T, 1)
                --> output shape (T, F)
        '''
        # using Taylor expansion for large alpha (hence beta) to better compute
        # (1 + u/beta)**(-alpha) 
        idx = np.logical_and(alpha < 1e10, beta < 1e10).ravel()
        if alpha.size == self.L:
            expect = np.empty_like(U)
            expect[idx] = (1 + U[idx]/beta[idx])**(-alpha[idx])
            expect[-idx] = np.exp(-U[-idx] * alpha[-idx]/beta[-idx])
        elif alpha.size == self.T:
            expect = np.empty((self.T, self.F))
            expect[idx] = (1 + U/beta[idx])**(-alpha[idx])
            expect[-idx] = np.exp(-U * alpha[-idx]/beta[-idx])
        else:
            raise ValueError('wrong dimension')
        expect[U <= -beta] = np.inf
        return expect

def print_gradient(name, val, grad, approx):
    print('{} = {:.2f}\tGradient: {:.2f}\tApprox: {:.2f}\t'
            '| Diff |: {:.3f}'.format(name, val, grad, approx, 
                np.abs(grad - approx)))

def comp_expect(alpha, beta):
    return (alpha/beta, special.psi(alpha) - np.log(beta))

def entropy(alpha, beta):
    return (alpha - np.log(beta) + special.gammaln(alpha) + 
            (1-alpha) * special.psi(alpha))

def approx_grad(f, x, delta=1e-8, args=()):
    x = np.asarray(x).ravel()
    grad = np.zeros_like(x)
    diff = delta * np.eye(x.size)
    for i in xrange(x.size):
        grad[i] = (f(x + diff[i], *args) - f(x - diff[i], *args)) / (2*delta)
    return grad

