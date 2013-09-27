"""

Beta-divergence NMF

Translate from MATLAB code by Minje Kim <minje@illinois.edu>
CREATED: 2013-09-16 03:04:33 by Dawen Liang <dl2771@columbia.edu>

"""

import numpy as np

eps = np.spacing(1)

def NMF_beta(X, K, maxiter=100, criterion=0.0001, W=None, beta=1, seed=None,
             normalize=False, verbose=True):
    ''' Beta-divergence NMF
    '''
    f, t = X.shape

    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if W is None:
        W = np.random.rand(f, K)
        updateW = True
    else:
        if K != W.shape[1]:
            raise ValueError('K != W.shape[1]')
        updateW = False

    H = np.random.rand(K, t)
    score = -np.inf

    if beta == 2:
        # EUC-NMF
        for i in xrange(maxiter):
            if updateW:
                W = W * np.dot(X, H.T)
                W = W / (np.dot(np.dot(W, H), H.T) + eps)
            H = H * np.dot(W.T, X)
            H = H / (np.dot(np.dot(W.T, W), H) + eps)
            if normalize:
                _normalize(W, H)

            lastscore = score
            X_bar = np.dot(W, H)
            score = np.sum((X - X_bar)**2)
            improvement = (lastscore - score) / abs(lastscore)
            if verbose:
                print ('iteration {}: obj = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
            if i >= 10 and improvement < criterion:
                break

    elif beta == 1:
        # KL-NMF
        for i in xrange(maxiter):
            if updateW:
                W = W * np.dot(X / (np.dot(W, H) + eps), H.T)
                W = W / (np.dot(np.ones((f, t)), H.T) + eps)
            H = H * np.dot(W.T, X / (np.dot(W, H) + eps))
            H = H / (np.dot(W.T, np.ones((f, t))) + eps)
            if normalize:
                _normalize(W, H)

            lastscore = score
            X_bar = np.dot(W, H)
            score = np.sum(X * (np.log(X) - np.log(X_bar)) - np.log(X) + np.log(X_bar))
            improvement = (lastscore - score) / abs(lastscore)
            if verbose:
                print ('iteration {}: obj = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
            if i >= 10 and improvement < criterion:
                break

    elif beta == 0:
        # IS-NMF
        for i in xrange(maxiter):
            if updateW:
                W = W * np.dot(X / (np.dot(W, H) + eps)**2, H.T)
                W = W / (np.dot((np.dot(W, H) + eps)**(-1), H.T) + eps)
            H = H * np.dot(W.T, X / (np.dot(W, H) + eps)**2)
            H = H / (np.dot(W.T, (np.dot(W, H) + eps)**(-1)) + eps)
            if normalize:
                _normalize(W, H)

            lastscore = score
            X_bar = np.dot(W, H)
            score = np.sum(X / X_bar + np.log(X) - np.log(X_bar) - 1)
            improvement = (lastscore - score) / abs(lastscore)
            if verbose:
                print ('iteration {}: obj = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
            if i >= 10 and improvement < criterion:
                break
    else:
        raise ValueError('beta can only be 0, 1, or 2')

    return (W, H)

def _normalize(W, H):
    scale = np.sqrt(np.sum(W**2, axis=0, keepdims=True))
    W = W / scale
    H = H * scale.T
