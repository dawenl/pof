import os
import sys

import numpy as np
import scipy.io as sio

import gamma_gvpl as vpl


def vlbound_spk_id(data_mat, L):
    d = sio.loadmat(data_mat)
    X_train, y_train = d['X_train'], d['y_train'].ravel()
    X_test, y_test = d['X_test'], d['y_test'].ravel()

    F = X_train.shape[0]
    n_spk = len(np.unique(y_train))

    U = np.zeros((n_spk, L, F))
    alpha = np.zeros((n_spk, L))
    gamma = np.zeros((n_spk, F))

    for spk in xrange(n_spk):
        X = X_train[:, y_train == spk]
        print('Learning prior for spk #{},'
              ' length = {} frames'.format(spk, X.shape[1]))
        threshold = 0.0005
        old_obj = -np.inf
        maxiter = 200
        cold_start = False
        batch_m = True

        sfd = vpl.SF_Dict(X.T, L=L, seed=98765)
        obj = []
        for i in xrange(maxiter):
            sfd.vb_e(cold_start=cold_start, disp=0)
            sfd.vb_m(batch=batch_m, disp=0)
            score = sfd.bound()
            obj.append(score)
            improvement = (score - old_obj) / abs(old_obj)
            print('After ITERATION: {}\tObjective: {:.2f}\t'
                  'Old objective: {:.2f}\t'
                  'Improvement: {:.4f}'.format(i, score, old_obj, improvement))
            if improvement < threshold:
                break
            old_obj = score
        U[spk] = sfd.U
        alpha[spk] = sfd.alpha
        gamma[spk] = sfd.gamma

    vlbound = np.zeros((n_spk, n_spk))
    for pspk in xrange(n_spk):
        for spk in xrange(n_spk):
            X = X_test[:, y_test == spk]
            print('Fit data from spk #{} with prior spk #{}, '
                  'length = {} frames'.format(spk, pspk, X.shape[1]))
            sfd = vpl.SF_Dict(X.T, L=L, seed=98765)
            sfd.U, sfd.alpha, sfd.gamma = U[pspk], alpha[pspk], gamma[pspk]
            sfd.vb_e(cold_start=cold_start, disp=0)
            vlbound[pspk, spk] = sfd.bound() / X.shape[1]
            print('Average variational '
                  'lower bound: {:.2f}'.format(vlbound[pspk, spk]))

    sio.savemat('spk_vlbound_L{}_{}'.format(L, data_mat), {'vlbound': vlbound})
    pass


if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    data_mat = sys.argv[1]
    L = int(sys.argv[2])
    vlbound_spk_id(data_mat, L)
    pass
