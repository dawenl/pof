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
seed = 357
np.random.seed(seed)
U = np.random.randn(F, L)
alpha = np.random.gamma(2, size=(L,))
gamma = np.random.gamma(100, 1./10, size=(F,))
A = np.empty((L, N))
for n in xrange(N):
    A[:,n] = np.random.gamma(alpha, scale=1./alpha)
V = np.dot(U, A) + np.random.normal(scale=np.sqrt(1./gamma)).reshape(F,1)
W = np.exp(W)

# <codecell>

subplot(311)
specshow(U)
colorbar()
subplot(312)
specshow(A)
colorbar()
subplot(313)
specshow(V)
colorbar()

# <codecell>

reload(dict_prior)
sfd = dict_prior.SF_Dict(W, L=10)
sfd.vb_e()

# <codecell>

gamma

# <codecell>

A

# <codecell>

amax(U)

# <codecell>

amax(dot(U, A))

# <codecell>


