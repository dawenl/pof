"""

Beta-divergence NMF with multiplicative updates

Translate from MATLAB code by Minje Kim <minje@illinois.edu>
CREATED: 2013-09-16 03:04:33 by Dawen Liang <dliang@columbia.edu>

"""

import numpy as np

eps = np.spacing(1)


def NMF_beta(X, K, W=None, beta=None, maxiter=500, tol=None, seed=None,
             normalize=False, verbose=False):
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

    if not beta in [0, 1, 2]:
        raise ValueError('beta has to be 0, 1 or 2')
    elif beta == 2:
        # EUC-NMF
        for i in xrange(maxiter):
            if updateW:
                X_bar = W.dot(H)
                W = W * X.dot(H.T)
                W = W / (X_bar.dot(H.T) + eps)
            H = H * W.T.dot(X) / (W.T.dot(W).dot(H) + eps)
            if normalize:
                _normalize(W, H)

            lastscore = score
            score = _compute_loss(X, W, H, beta)
            improvement = (lastscore - score) / abs(lastscore)
            if verbose:
                print ('Iteration %d: obj = %.2f (%.5f improvement)' %
                       (i, score, improvement))
            if i >= 10 and improvement < tol:
                break
    elif beta == 1:
        # KL-NMF
        for i in xrange(maxiter):
            if updateW:
                X_bar = W.dot(H)
                W = W * np.dot(X / (X_bar + eps), H.T)
                W = W / (np.dot(np.ones((f, t)), H.T) + eps)
            X_bar = W.dot(H)
            H = H * W.T.dot(X / (X_bar + eps))
            H = H / (W.T.dot(np.ones((f, t))) + eps)
            if normalize:
                (W, H) = _normalize(W, H)

            lastscore = score
            score = _compute_loss(X, W, H, beta)
            improvement = (lastscore - score) / abs(lastscore)
            if verbose:
                print ('Iteration %d: obj = %.2f (%.5f improvement)' %
                       (i, score, improvement))
            if improvement < tol:
                break
    elif beta == 0:
        # IS-NMF
        for i in xrange(maxiter):
            if updateW:
                X_bar = W.dot(H)
                W = W * np.dot(X / (X_bar + eps)**2, H.T)
                W = W / (np.dot((X_bar + eps)**(-1), H.T) + eps)
            X_bar = W.dot(H)
            H = H * W.T.dot(X / (X_bar + eps)**2)
            H = H / (W.T.dot((X_bar + eps)**(-1)) + eps)
            if normalize:
                (W, H) = _normalize(W, H)

            lastscore = score
            score = _compute_loss(X, W, H, beta)
            improvement = (lastscore - score) / abs(lastscore)
            if verbose:
                print ('Iteration %d: obj = %.2f (%.5f improvement)' %
                       (i, score, improvement))
            if improvement < tol:
                break
    return (W, H)


def _normalize(W, H):
    scale = np.sqrt(np.sum(W**2, axis=0, keepdims=True))
    W = W / scale
    H = H * scale.T
    return (W, H)


def _compute_loss(X, W, H, beta):
    loss = np.inf
    f, t = X.shape
    X_bar = W.dot(H)
    if beta == 0:
        loss = np.sum(X / X_bar - np.log(X) + np.log(X_bar)) - f * t
    elif beta == 1:
        loss = np.sum(X * (np.log(X) - np.log(X_bar)) - X + X_bar)
    elif beta == 2:
        loss = np.sum((X - X_bar)**2)
    return loss
