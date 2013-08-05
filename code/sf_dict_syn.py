# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools

from matplotlib.pyplot import *

import numpy as np
import gvpl as vpl

# <codecell>

fig = functools.partial(figure, figsize=(16, 4))
specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

# <codecell>

# Synthetic data
F = 257
L = 20
T = 80
seed = 3579
np.random.seed(seed)
U = np.random.randn(L, F)
alpha = np.random.gamma(1, size=(L,))
gamma = np.random.gamma(100, 1./10, size=(F,))
A = np.empty((T, L))
V = np.empty((T, F))
for t in xrange(T):
    A[t,:] = np.random.gamma(alpha, scale=1./alpha)
    V[t,:] = np.dot(A[t,:], U) + np.random.normal(scale=np.sqrt(1./gamma))
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

reload(vpl)
threshold = 0.005
old_obj = -np.inf
maxiter = 50
cold_start = False
batch_m = False

sfd = vpl.SF_Dict(W, L=L, seed=98765)
obj = []

for i in xrange(maxiter):
    sfd.vb_e(cold_start=cold_start, disp=1)
    sfd.vb_m(batch=batch_m, disp=1)
    score = sfd.bound()
    obj.append(score)
    improvement = (score - old_obj) / abs(old_obj)
    print 'After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.4f}'.format(i, score, old_obj, improvement)
    if improvement < threshold:
        break
    old_obj = score

# <codecell>

plot(obj)
pass

# <codecell>

b = sfd.a / sfd.mu
fig()
mab = max(amax(sfd.a), amax(b))
bins = linspace(0, mab, 100)
hist(sfd.a.ravel(), bins, alpha=0.5)
hist(b.ravel(), bins, alpha=0.5)
legend(["alpha", "beta"])
fig()
hist(sfd.EA.ravel(), bins=100)
print(amax(sfd.a), amax(b))
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
    tight_layout()
    
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

