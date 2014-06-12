# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import os
import sys
import scipy.io as sio

import npof


def train_pof(data_file, n_filters, n_jobs):
    d = sio.loadmat(data_file)
    W = d['W']

    coder = npof.ProductOfFiltersLearning(n_feats=W.shape[0],
                                          n_filters=n_filters,
                                          n_jobs=n_jobs,
                                          tol=0.0001,
                                          save_filters=True,
                                          random_state=98765,
                                          verbose=True)
    coder.fit(W.T)
    pass


if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    data_file = sys.argv[1]
    n_filters = int(sys.argv[2])
    n_jobs = int(sys.argv[3])
    print('Learn PoF with n_filters=%d ond n_jobs=%d on %s...' % (
        n_filters, n_jobs, data_file))
    train_pof(data_file, n_filters, n_jobs)
    pass
