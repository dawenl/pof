# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools

from matplotlib.pyplot import *

import numpy as np
import dict_prior as dp

# <codecell>

specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

# <codecell>

# Synthetic data
F = 50
L = 10
T = 50
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

# <codecell>

subplot(311)
specshow(U.T)
title('U')
colorbar()
subplot(312)
specshow(A.T)
title('A')
colorbar()
subplot(313)
specshow(V.T)
title('log(W)')
colorbar()
tight_layout()
pass

# <codecell>

threshold = 0.01
old_obj = -np.inf
maxiter = 100
cold_start = True

sfd = dp.SF_Dict(W, L=4*L, seed=98765)
obj = []
for i in xrange(maxiter):
    sfd.vb_e(cold_start=cold_start, disp=1)
    if sfd.vb_m(disp=1):
        break
    obj.append(sfd.obj)
    improvement = (sfd.obj - old_obj) / abs(sfd.obj)
    print 'After ITERATION: {}\tImprovement: {:.4f}'.format(i, improvement)
    if (sfd.obj - old_obj) / abs(sfd.obj) < threshold:
        break
    old_obj = sfd.obj

# <codecell>

plot(obj)
pass

# <codecell>

idx_alpha_sfd = np.flipud(argsort(sfd.alpha))
idx_alpha = np.flipud(argsort(alpha))

plot(sfd.alpha[idx_alpha_sfd], '-o')
plot(alpha[idx_alpha], '-*')
pass

# <codecell>

def normalize_and_plot(A, U, normalize=False):
    if normalize:
        tmpA = A / np.max(A, axis=0, keepdims=True)
        tmpU = U * np.max(A, axis=0, keepdims=True).T  
    else:
        tmpA = A
        tmpU = U
    figure()
    subplot(211)
    specshow(tmpA.T)
    title('A')
    colorbar()
    subplot(212)
    specshow(tmpU.T)
    title('U')
    colorbar()
    
normalize_and_plot(sfd.EA[:,idx_alpha_sfd], sfd.U[idx_alpha_sfd,:])
normalize_and_plot(A[:,idx_alpha], U[idx_alpha,:])

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


