import os
import sys
import numpy as np
import scipy.io as sio

import gamma_gvpl as vpl


def train_sf(W, L, threshold=0.0005, maxiter=200):
    old_obj = -np.inf
    cold_start = False
    batch_m = True

    sfd = vpl.SF_Dict(W.T, L=L, seed=98765)
    for i in xrange(maxiter):
        sfd.vb_e(cold_start=cold_start, disp=0)
        sfd.vb_m(batch=batch_m, disp=0)
        score = sfd.bound()
        improvement = (score - old_obj) / abs(old_obj)
        print('After ITERATION: {}\tObjective: {:.2f}\t''Old objective: '
              '{:.2f}\tImprovement: {:.4f}'.format(i, score, old_obj,
                                                   improvement))
        if improvement < threshold:
            break
        old_obj = score
    sio.savemat('sf_prior_L{}.mat'.format(L),
                {'U': sfd.U, 'alpha': sfd.alpha, 'gamma': sfd.gamma})
    pass

if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    L = int(sys.argv[1])
    d = sio.loadmat('TIMIT_subset.mat')
    W = d['W']
    print('Train SF prior with L={} on TIMIT subset...'.format(L))
    train_sf(W, L)
    pass
