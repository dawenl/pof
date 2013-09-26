import numpy as np
import glob
import time
import sys
import os

import scipy.io as sio

import librosa
import kl_nmf


def train(nmf, updateW=True, criterion=0.0005, maxiter=1000):
    score = nmf.bound()
    objs = []
    for i in xrange(maxiter):
        start_t = time.time()
        nmf.update(updateW=updateW, disp=1)
        t = time.time() - start_t

        lastscore = score
        score = nmf.bound()
        objs.append(score)
        improvement = (score - lastscore) / abs(lastscore)
        print ('iteration {}: bound = {:.2f} ({:.5f} improvement) '
               'time = {:.2f}'.format(i, score, improvement, t))
        if i > 10 and improvement < criterion:
            break
    return objs


def learn_dictionary(prior_mat, K, d, n_fft=1024, hop_length=512, seed=None):
    prior = sio.loadmat(prior_mat)
    U = prior['U'].T
    gamma = prior['gamma']
    alpha = prior['alpha'].ravel()

    paths = sorted(glob.glob('denoise/TIMIT_speech/*.wav'))

    n = len(paths)
    W_nu = np.zeros((n, n_fft/2 + 1, K))
    W_rho = np.zeros_like(W_nu)

    for (i, path) in enumerate(paths):
        print('Learn dictionary for {}...'.format(path))
        x, sr = librosa.load(path, sr=None)
        X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))

        nmf_sf = kl_nmf.KL_NMF(X, K=K, d=d, seed=seed,
                               U=U, gamma=gamma, alpha=alpha)
        train(nmf_sf)
        W_nu[i], W_rho[i] = nmf_sf.nuw, nmf_sf.rhow

    sio.savemat('SF_TIMIT60_dict_{}_K{}_d{}.mat'.format(prior_mat, K, d),
                {'W_nu': W_nu, 'W_rho': W_rho})
    return


if __name__ == '__main__':
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    prior_mat = sys.argv[1]
    K = int(sys.argv[2])
    d = int(sys.argv[3])
    learn_dictionary(prior_mat, K, d, seed=98765)
    pass
