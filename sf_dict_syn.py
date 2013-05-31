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
pass

# <codecell>

reload(dict_prior)
sfd = dict_prior.SF_Dict(W, L=L, seed=123)

# <codecell>

obj = []
for i in xrange(2):
    while not sfd.vb_e():
        print 'Bad initialization, restart'
    sfd.vb_m()
    obj.append(sfd.obj)

# <codecell>

plot(obj)

# <codecell>

def normalize_and_plot(A, U):
    tmpA = A / np.max(A, axis=0, keepdims=True)
    tmpU = U * np.max(A, axis=0, keepdims=True).T
    
    figure()
    subplot(211)
    specshow(tmpA.T)
    title('A')
    colorbar()
    subplot(212)
    specshow(tmpU.T)
    title('U')
    colorbar()
    
normalize_and_plot(sfd.EA, sfd.U)
normalize_and_plot(A, U)

# <codecell>

plot(sort(sfd.alpha), '-o')
plot(sort(alpha), '-*')

# <codecell>

V_rec = np.dot(sfd.EA, sfd.U)
subplot(311)
specshow(V.T)
colorbar()
subplot(312)
specshow(V_rec.T)
colorbar()
subplot(313)
specshow((V - V_rec).T)
colorbar()
pass

# <codecell>


