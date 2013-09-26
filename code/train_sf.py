import os
import sys
import numpy as np
import scipy.io as sio

import gamma_gvpl as vpl


def train_sf(matfile, L, threshold=0.0005, maxiter=200):
    d = sio.loadmat(matfile)
    W = d['W']

    old_obj = -np.inf
    cold_start = False
    batch_m = True

    sfd = vpl.SF_Dict(W.T, L=L, seed=98765)
    for i in xrange(maxiter):
        sfd.vb_e(cold_start=cold_start, disp=1)
        sfd.vb_m(batch=batch_m, disp=1)
        score = sfd.bound()
        improvement = (score - old_obj) / abs(old_obj)
        print('After ITERATION: {}\tObjective: {:.2f}\t''Old objective: '
              '{:.2f}\tImprovement: {:.4f}'.format(i, score, old_obj,
                                                   improvement))
        if improvement < threshold:
            break
        old_obj = score
    sio.savemat('sf_L{}_{}'.format(L, matfile),
                {'U': sfd.U, 'alpha': sfd.alpha, 'gamma': sfd.gamma})
    pass

if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    matfile = sys.argv[1]
    L = int(sys.argv[2])
    print('Train SF prior with L={} on {}...'.format(L, matfile))
    train_sf(matfile, L)
    pass
