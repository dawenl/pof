# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
import sys
import numpy as np
import scipy.io as sio

import gamma_gvpl as vpl

# <codecell>

def train_sf(matfile, L, threshold=0.0005, maxiter=200):
    d = sio.loadmat(matfile)
    W = d['W']

    old_obj = -np.inf

    sfd = vpl.SF_Dict(W.T, L=L, seed=98765)
    for i in xrange(maxiter):
        sfd.vb_e(disp=1)
        sfd.vb_m(disp=1)
        score = sfd.bound()
        improvement = (score - old_obj) / abs(old_obj)
        print('After ITERATION: {}\tObjective: {:.2f}\t''Old objective: '
              '{:.2f}\tImprovement: {:.5f}'.format(i, score, old_obj,
                                                   improvement))
        sio.savemat('sf_L{}_{}.iter{}'.format(L, matfile, i),
                    {'U': sfd.U, 'alpha': sfd.alpha, 'gamma': sfd.gamma})
        if improvement < threshold:
            break
        old_obj = score
    pass

# <codecell>

train_sf('TIMIT_spk30_F400_H160', 50)

# <codecell>

if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    matfile = sys.argv[1]
    L = int(sys.argv[2])
    print('Train SF prior with L={} on {}...'.format(L, matfile))
    train_sf(matfile, L)
    pass

