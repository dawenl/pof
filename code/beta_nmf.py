"""

Beta-divergence NMF

Translate from MATLAB code by Minje Kim <minje@illinois.edu>
CREATED: 2013-09-16 03:04:33 by Dawen Liang <dl2771@columbia.edu>

"""

import numpy as np

eps = np.spacing(1)

def NMF_beta(X, K, niter=100, W=None, beta=1, seed=None, normalize=False):
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

    if beta == 2:
        # EUC-NMF
        for _ in xrange(niter):
            if updateW:
                W = W * np.dot(X, H.T)
                W = W / (np.dot(np.dot(W, H), H.T) + eps)
            H = H * np.dot(W.T, X)
            H = H / (np.dot(np.dot(W.T, W), H) + eps)
    elif beta == 1:
        # KL-NMF
        for _ in xrange(niter):
            if updateW:
                W = W * np.dot(X / (np.dot(W, H) + eps), H.T)
                W = W / (np.dot(np.ones((f, t)), H.T) + eps)
            H = H * np.dot(W.T, X / (np.dot(W, H) + eps))
            H = H / (np.dot(W.T, np.ones((f, t))) + eps)
            if normalize:
                _normalize(W, H)
    elif beta == 0:
        # IS-NMF
        for _ in xrange(niter):
            if updateW:
                W = W * np.dot(X / (np.dot(W, H) + eps)**2, H.T)
                W = W / (np.dot((np.dot(W, H) + eps)**(-1), H.T) + eps)
            H = H * np.dot(W.T, X / (np.dot(W, H) + eps)**2)
            H = H / (np.dot(W.T, (np.dot(W, H) + eps)**(-1)) + eps)
            if normalize:
                _normalize(W, H)
    else:
        raise ValueError('beta can only be 0, 1, or 2')

    return (W, H)

def _normalize(W, H):
    scale = np.sqrt(np.sum(W**2, axis=0, keepdims=True))
    W = W / scale
    H = H * scale.T
