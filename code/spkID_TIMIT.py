'''
Load in the pre-generated MFCC or PoFC from cPickle file and learn Random
Forest with different train/test split

CREATED: 2014-02-15 02:50:39 by Dawen Liang <dliang@ee.columbia.edu>

'''


import os
import sys
import time

import cPickle as pickle
import numpy as np

from sklearn import ensemble


SENTS_PER_SPK = 10


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in xrange(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def permute_data(X_test, y_test, loc_test):
    """Permute the data given the location
    """
    X_test_rand = np.zeros_like(X_test)
    y_test_rand = np.zeros_like(y_test)

    start_p = 0
    for loc in loc_test:
        X_test_rand[:, start_p: start_p + (loc[1] - loc[0])] = \
            X_test[:, loc[0]: loc[1]]
        y_test_rand[start_p: start_p + (loc[1] - loc[0])] = \
            y_test[loc[0]: loc[1]]
        start_p += (loc[1] - loc[0])
    return (X_test_rand, y_test_rand)


def smooth(y_test, y_pred, max_k=200):
    """Apply a median filter smoother to the raw prediction
    """
    max_acc_med = 0
    for k in xrange(1, max_k, 2):
        y_pred_med = medfilt(y_pred, k)
        acc_med = np.sum(y_test == y_pred_med) / float(y_test.size)
        if max_acc_med < acc_med:
            max_acc_med = acc_med
            best_k = k
    print 'Raw acc: %.3f' % (np.sum(y_test == y_pred) / float(y_test.size))
    print 'Smoothed acc (k = %d): %.3f' % (best_k, max_acc_med)


def majority_voting(y_test, y_pred, loc, n_class):
    """Aggregate the frame-level predictions to a sentence-level prediction
    by majority voting
    """
    acc = 0
    idx = np.cumsum(np.diff(loc))
    for (start, end) in zip(idx[:-1], idx[1:]):
        major = np.argmax(np.bincount(y_pred[start: end], minlength=n_class))
        assert np.unique(y_test[start: end]).size == 1
        acc += (y_test[start] == major)
    return float(acc) / loc.shape[0]


def train_RF(feats, is_POFC, n_spk, n_train, n_jobs, n_estimators=200,
             rand_permute=True):
    """Generating train/test split, train Random Forest, report accuracy
    """

    n_test = SENTS_PER_SPK - n_train

    X_train, X_test = None, None
    y_train, y_test = None, None

    loc_test = np.zeros((n_spk * n_test, 2), dtype=np.int32)

    print 'Generating training/test split...'
    start_t = time.time()
    for i in xrange(n_spk):
        for j in xrange(n_train):
            feat = feats[SENTS_PER_SPK * i + j]
            if is_PoFC:
                feat = feat.T
            if X_train is None:
                X_train = feat
                y_train = i * np.ones((feat.shape[1], ), dtype=np.int32)
            else:
                X_train = np.hstack((X_train, feat))
                y_train = np.hstack((y_train, i * np.ones((feat.shape[1], ),
                                                          dtype=np.int32)))

        for j in xrange(n_train, SENTS_PER_SPK):
            feat = feats[SENTS_PER_SPK * i + j]
            if is_PoFC:
                feat = feat.T
            if X_test is None:
                X_test = feat
                y_test = i * np.ones((feat.shape[1], ), dtype=np.int32)
                loc_test[i * n_test + (j - n_train), 0] = 0
                loc_test[i * n_test + (j - n_train), 1] = X_test.shape[1]
            else:
                loc_test[i * n_test + (j - n_train), 0] = X_test.shape[1]
                X_test = np.hstack((X_test, feat))
                y_test = np.hstack((y_test, i * np.ones((feat.shape[1], ),
                                                        dtype=np.int32)))
                loc_test[i * n_test + (j - n_train), 1] = X_test.shape[1]

    print 'Generating finished\t# train: %d, # test: %d\ttime: %.2f' % \
        (X_train.shape[1], X_test.shape[1], time.time() - start_t)

    np.random.seed(98765)
    if rand_permute:
        print 'Permuate test data...'
        loc_test = np.random.permutation(loc_test)
    X_test, y_test = permute_data(X_test, y_test, loc_test)

    print 'Training Random Forest...'
    start_t = time.time()
    clf = ensemble.RandomForestClassifier(n_estimators=n_estimators,
                                          min_samples_split=50,
                                          min_samples_leaf=25,
                                          n_jobs=n_jobs,
                                          verbose=5)
    clf.fit(X_train.T, y_train)
    print 'Modeling training finished\ttime: %.2f' % (time.time() - start_t)

    print 'Predict on the test data...'
    y_pred = clf.predict(X_test.T)
    smooth(y_test, y_pred, max_k=30)

    acc_mv = majority_voting(y_test, y_pred, loc_test, n_spk)
    print 'Majority voting: %.3f\n' % acc_mv
    pass


if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    data_file = sys.argv[1]
    n_spk = int(sys.argv[2])
    n_jobs = int(sys.argv[3])

    print 'Loading data from cPickle file...'
    start_t = time.time()
    with open(data_file, 'rb') as fp:
        feats = pickle.load(fp)
    print 'Loading finished\ttime: %.2f' % (time.time() - start_t)
    is_PoFC = ('pofc' in data_file)

    for n_train in xrange(1, SENTS_PER_SPK):
        print('*************************************************************\n'
              '***%s on %d/%d train/test split with %d speakers***\n'
              '*************************************************************'
              % (data_file, n_train, SENTS_PER_SPK - n_train, n_spk))
        train_RF(feats, is_PoFC, n_spk, n_train, n_jobs)
    pass
