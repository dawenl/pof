'''
CREATED: 2013-05-23 10:58:16 by Dawen Liang <daliang@adobe.com>
'''

import numpy as np
import scipy.optimize as optimize

class SF_Dict:
    def __init__(self, W, L=10, smoothness=100, seed=None, **kwargs):
        self.V = np.log(W)
        self.F, self.N = W.shape
        self.L = L
        if seed is None:
            print 'Using random seed'
            seed = np.random.seed()
        self._parse_args(**kwargs)
        self._init(smoothness=smoothness)

    def _parse_args(self, **kwargs):
        pass

    def _init(self, smoothness=100):
        # model parameters
        self.U = np.random.randn(self.F, self.L)
        self.alpha = np.random.gamma(smoothness, 1./smoothness, size=(self.L,))
        self.gamma = np.random.gamma(smoothness, 1./smoothness, size=(self.F,))

        # variational parameters and expectations
        self._init_variational(smoothness)

    def _init_variational(self, smoothness):
        # just in case seed was fixed previously
        np.random.seed()
        self.mu = np.random.randn(self.L, self.N)
        self.r = np.random.gamma(smoothness, 1./smoothness, size=(self.L,
            self.N))
        self.EA, self.EA2 = self._comp_expect(self.mu, self.r)
        self.old_mu, self.old_r = None, None

    def _comp_expect(self, mu, r):
        return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r))
         
    #def update(self, e_converge=True, smoothness=100, fmin='LBFGS', verbose=True):
    #    self.vb_e(e_converge, smoothness, fmin, verbose)
    #    self.vb_m(fmin, verbose)
    #    self._objective()

    def vb_e(self, e_converge=True, smoothness=100, fmin='LBFGS', verbose=True):
        if e_converge:
            # do e-step until variational inference converges
            self._init_variational(smoothness)
            while True:
                for l in xrange(self.L):
                    if not self.update_phi(l, fmin):
                        return False
                if self.old_mu is None:
                    self.old_mu = self.mu.copy()
                    self.old_r = self.r.copy()
                else:
                    if np.sum(np.abs(self.old_mu - self.mu)) <= 1e-4 and np.sum(np.abs(self.old_r - self.r)) <= 1e-4:
                        break
        else:
            # do e-step for one iteration
            for l in xrange(self.L):
                if not self.update_phi(l, fmin):
                    return False
        return True

    def update_phi(self, l, fmin):                
        def f_stub(phi):
            lcoef = np.sum(self.gamma.reshape(-1, 1) * np.outer(self.U[:,l], np.exp(phi)) * Eres, axis=0) 
            qcoef = -1./2 * np.sum(self.gamma.reshape(-1, 1) * np.outer(self.U[:,l]**2, np.exp(2*phi)), axis=0) 
            return (lcoef, qcoef)

        def f(phi):
            const = self.alpha[l] * (phi - np.exp(phi))   
            lcoef, qcoef = f_stub(phi)
            return -np.sum(const + lcoef + qcoef)

        def df(phi):
            const = self.alpha[l] * (1 - np.exp(phi))
            lcoef, qcoef = f_stub(phi)
            return -(const + lcoef + 2*qcoef)
            
        def df2(phi):
            const = -self.alpha[l] * np.exp(phi)
            lcoef, qcoef = f_stub(phi)
            return -(const + lcoef + 4*qcoef)

        Eres = self.V - np.dot(self.U, self.EA) + np.outer(self.U[:,l],
                self.EA[l,:])
        phi0 = self.mu[l,:]
        if fmin == 'LBFGS':
            mu_hat, _, d = optimize.fmin_l_bfgs_b(f, phi0, fprime=df, disp=0)
        elif fmin == 'NCG':
            def _fhess(phi):
                return np.diag(df2(phi))
            mu_hat = optimize.fmin_ncg(f, phi0, df, fhess=_fhess)
        else:
            raise ValueError('fmin is either LBFGS or NCG')

        self.mu[l,:], self.r[l,:] = mu_hat, df2(mu_hat)
        if np.any(self.r[l,:] <= 0):
            if fmin == 'LBFGS':
                if d['warnflag'] == 2:
                    print 'A[{}, :]: {}, f={}'.format(l, d['task'], f(mu_hat))
                else:
                    print 'A[{}, :]: {}, f={}'.format(l, d['warnflag'], f(mu_hat))
            return False 

        self.EA[l,:], self.EA2[l,:] = self._comp_expect(self.mu[l,:],
                self.r[l,:])
        return True

    def vb_m(self, fmin, verbose):
        if verbose:
            print 'Updating U...'
        for l in xrange(self.L):
            self.update_u(l, fmin)
        if verbose:
            print 'Updating gamma and alpha'
        self.update_gamma(fmin)
        self.update_alpha(fmin)
        pass

    def update_u(self, l, fmin):
        pass

    def update_gamma(self, fmin):
        pass

    def update_alpha(self, fmin):
        pass

    def _objective(self):
        pass
