"""

Stochastic source-filter prior dictionary learning

CREATED: 2013-07-08 11:06:12 by Dawen Liang <daliang@adobe.com> 

"""

import numpy as np
import scipy.special as special

import gvpl as vpl

class SF_Dict(vpl.SF_Dict):
    def __init__(self, W, L=10, smoothness=100, seed=None):
        super(SF_Dict, self).__init__(W, L=L, smoothness=smoothness, seed=seed)

    def switch(self, W, smoothness=100):
        self.V = np.log(W)
            
    def vb_m(self, rho, batch=False, atol=0.01, verbose=True, disp=0, update_alpha=True):
        print 'Variational M-step...'
        if batch:
            self.update_u_batch(rho, disp)
        else:
            for l in xrange(self.L):
                self.update_u(l, rho, disp)
        self.update_gamma(rho)
        if update_alpha:
            self.update_alpha(rho, disp)
        self._objective()
        return False

    def update_u_batch(self, rho, disp):
        def df(u):
            U = u.reshape(self.L, self.F)
            grad_U = np.zeros_like(U)
            for l in xrange(self.L):
                Eres = self.V - np.dot(self.EA, U) + np.outer(self.EA[:,l], U[l,:])
                grad_U[l,:] = np.mean(np.outer(self.EA2[:,l], U[l,:]) - Eres * self.EA[:,l][np.newaxis].T, axis=0)
            return grad_U.ravel()

        u = self.U.ravel()
        self.U = (u - rho * df(u)).reshape(self.L, self.F)

    def update_u(self, l, rho, disp):
        def df(u):
            return np.mean(np.outer(self.EA2[:,l], u) - Eres * self.EA[:,l][np.newaxis].T, axis=0)

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        self.U[l, :] -= rho * df(self.U[l,:])

    def update_gamma(self, rho):
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        new_gamma = 1./np.mean(self.V**2 - 2 * self.V * EV + EV2, axis=0)
        self.gamma = (1 - rho) * self.gamma + rho * new_gamma

    def update_alpha(self, rho, disp):
        def df(eta):
            return -np.exp(eta) * ((eta + 1 - special.psi(np.exp(eta))) + np.mean(self.ElogA - self.EA, axis=0))
        
        eta = np.log(self.alpha) - rho * df(np.log(self.alpha))
        self.alpha = np.exp(eta)

