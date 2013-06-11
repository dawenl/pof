import numpy as np
import dict_prior

F = 32
L = 20 
T = 500
seed = 3579
np.random.seed(seed)
U = np.random.randn(L, F)
alpha = np.random.gamma(1, size=(L,))
gamma = np.random.gamma(100, 1./10, size=(F,))
A = np.empty((T, L))
for t in xrange(T):
    A[t,:] = np.random.gamma(alpha, scale=1./alpha)
V = np.dot(A, U) + np.random.normal(scale=np.sqrt(1./gamma))
W = np.exp(V)

sfd = dict_prior.SF_Dict(W, L=L, seed=123)
maxiter = 5 
for i in xrange(maxiter):
    print 'ITERATION: {}'.format(i)
    sfd.vb_e()
    if sfd.vb_m():
        break
pass
