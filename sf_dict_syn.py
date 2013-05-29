# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools

from matplotlib.pyplot import *

import numpy as np
import dict_prior

# <codecell>

specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

# <codecell>

# Synthetic data
F = 128
L = 20
N = 500
seed = 3579
np.random.seed(seed)
U = np.random.randn(L, F)
alpha = np.random.gamma(2, size=(L,))
gamma = np.random.gamma(100, 1./10, size=(F,))
A = np.empty((N, L))
for n in xrange(N):
    A[n,:] = np.random.gamma(alpha, scale=1./alpha)
V = np.dot(A, U) + np.random.normal(scale=np.sqrt(1./gamma))
W = np.exp(V)

# <codecell>

subplot(311)
specshow(U.T)
colorbar()
subplot(312)
specshow(A.T)
colorbar()
subplot(313)
specshow(V.T)
colorbar()

# <codecell>

reload(dict_prior)
sfd = dict_prior.SF_Dict(W, L=L)

# <codecell>

for i in xrange(5):
    sfd.vb_e()
    sfd.vb_m()

# <codecell>

subplot(211)
specshow(sfd.EA.T)
colorbar()
subplot(212)
specshow(A.T)
colorbar()

