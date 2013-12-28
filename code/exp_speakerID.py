# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sklearn import svm, mixture
import functools, glob

import numpy as np
from scikits.audiolab import Sndfile, Format
from matplotlib.pyplot import *

import scipy.io as sio

import librosa
import librosa.filters

# <codecell>

TIMIT_DIR = '../../timit/test/'

# <codecell>

fig = functools.partial(figure, figsize=(16,4))
specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

# <codecell>

def logspec(X, amin=1e-10, dbdown=80):
    logX = 20 * np.log10(np.maximum(X, amin))
    return np.maximum(logX, logX.max() - dbdown)

def load_timit(wav_dir):
    f = Sndfile(wav_dir, 'r')
    wav = f.read_frames(f.nframes)
    return (wav, f.samplerate)

def write_wav(w, filename, channels=1, samplerate=16000):
    f_out = Sndfile(filename, 'w', format=Format(), channels=channels, samplerate=samplerate)
    f_out.write_frames(w)
    f_out.close()
    pass

# <codecell>

def z_score(X_train, X_test):
    meanX = np.mean(X_train, axis=1, keepdims=True)
    stdX = np.std(X_train, axis=1, keepdims=True)
    
    X_train = (X_train - meanX) / stdX
    X_test = (X_test - meanX) / stdX
    
    return (X_train, X_test)

def diff_feat(X):
    tmp = X.copy()
    L = X.shape[0]
    X = np.vstack((X, np.hstack((np.zeros((L, 1)), np.diff(tmp, n=1, axis=1)))))
    X = np.vstack((X, np.hstack((np.zeros((L, 2)), np.diff(tmp, n=2, axis=1)))))
    return X

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

# <codecell>

f_dirs_all = !ls -d "$TIMIT_DIR"dr[1-6]/f*
m_dirs_all = !ls -d "$TIMIT_DIR"dr[1-6]/m*

n_spk = 5
np.random.seed(98765)
f_dirs = np.random.permutation(f_dirs_all)[:n_spk]
m_dirs = np.random.permutation(m_dirs_all)[:n_spk]

files = [glob.glob(spk_dir + '/*.wav') for spk_dir in f_dirs]
files.extend([glob.glob(spk_dir + '/*.wav') for spk_dir in m_dirs])

# <codecell>

print f_dirs
print m_dirs

# <codecell>

N_train = 8

# <codecell>

n_fft = 1024
hop_length = 512

X_train_mfcc = None
X_test_mfcc = None
y_train = None
y_test = None

mfcc_d = 13

loc_test = np.zeros((2 * n_spk * (10 - N_train), 2))

for (i, spk_dir) in enumerate(files):
    for wav_dir in spk_dir[:N_train]:
        wav, sr = load_timit(wav_dir)
        S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
        #log_S = np.log(np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))
        if X_train_mfcc is None:
            X_train_mfcc = librosa.feature.mfcc(librosa.logamplitude(S), n_mfcc=mfcc_d)
            #X_train_mfcc = np.dot(librosa.filters.dct(mfcc_d, log_S.shape[0]), log_S)
            y_train = i * np.ones((X_train_mfcc.shape[1], ))
        else:
            mfcc = librosa.feature.mfcc(librosa.logamplitude(S), n_mfcc=mfcc_d)
            #mfcc = np.dot(librosa.filters.dct(mfcc_d, log_S.shape[0]), log_S)
            X_train_mfcc = np.hstack((X_train_mfcc, mfcc))
            y_train = np.hstack((y_train, i * np.ones((mfcc.shape[1], ))))
                      
    for (j, wav_dir) in enumerate(spk_dir[N_train:]):
        wav, sr = load_timit(wav_dir)
        S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
        #log_S = np.log(np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))
        if X_test_mfcc is None:
            X_test_mfcc = librosa.feature.mfcc(librosa.logamplitude(S), n_mfcc=mfcc_d)
            #X_test_mfcc = np.dot(librosa.filters.dct(mfcc_d, log_S.shape[0]), log_S)
            y_test = i * np.ones((X_test_mfcc.shape[1], )) 
            loc_test[i * (10 - N_train) + j, 0] = 0
            loc_test[i * (10 - N_train) + j, 1] = X_test_mfcc.shape[1]
        else:
            mfcc = librosa.feature.mfcc(librosa.logamplitude(S), n_mfcc=mfcc_d)
            #mfcc = np.dot(librosa.filters.dct(mfcc_d, log_S.shape[0]), log_S)
            loc_test[i * (10 - N_train) + j, 0] = X_test_mfcc.shape[1]
            X_test_mfcc = np.hstack((X_test_mfcc, mfcc))
            y_test = np.hstack((y_test, i * np.ones((mfcc.shape[1], ))))
            loc_test[i * (10 - N_train) + j, 1] = X_test_mfcc.shape[1]

# <codecell>

np.random.seed(98765)
loc_test_rand = np.random.permutation(loc_test)
print loc_test_rand

# <codecell>

def permute_data(X_test, y_test, loc_test): 
    X_test_rand = np.zeros_like(X_test)
    y_test_rand = np.zeros_like(y_test)
    
    start_p = 0
    for loc in loc_test:
        #print loc
        #print [start_p, start_p + (loc[1] - loc[0])]
        X_test_rand[:, start_p: start_p + (loc[1] - loc[0])] = X_test[:, loc[0] : loc[1]]
        y_test_rand[start_p: start_p + (loc[1] - loc[0])] = y_test[loc[0]: loc[1]]
        start_p += (loc[1] - loc[0])
    return (X_test_rand, y_test_rand)

# <codecell>

X_test_mfcc, y_test = permute_data(X_test_mfcc, y_test, loc_test_rand)

# <codecell>

X_train_mfcc, X_test_mfcc = z_score(X_train_mfcc, X_test_mfcc)
X_train_mfcc = diff_feat(X_train_mfcc)
X_test_mfcc = diff_feat(X_test_mfcc)

print X_train_mfcc.shape, X_test_mfcc.shape

# <codecell>

d = sio.loadmat('feats/feat_sf_L50_TIMIT_spk20_spkID_Train{}.mat'.format(N_train))
X_train_sf = d['A_train']
X_test_sf = d['A_test']

X_test_sf, _ = permute_data(X_test_sf, d['y_test'], loc_test_rand)

X_train_sf, X_test_sf = z_score(X_train_sf, X_test_sf)

X_train_sf = diff_feat(X_train_sf)
X_test_sf = diff_feat(X_test_sf)

print X_train_sf.shape, X_test_sf.shape

# <codecell>

fig(figsize=(12, 8))
subplot(221)
specshow(X_train_mfcc)
colorbar()
subplot(222)
specshow(X_test_mfcc)
colorbar()
subplot(223)
specshow(X_train_sf)
colorbar()
subplot(224)
specshow(X_test_sf)
colorbar()
pass

# <codecell>

clf = svm.LinearSVC()

def spk_id(clf, X_train, X_test, y_train):
    clf.fit(X_train.T, y_train)
    y_pred = clf.predict(X_test.T)
    return y_pred

def smooth(y_test, y_pred, max_k = 200):
    max_acc_med = 0
    for k in xrange(1, max_k, 2):
        y_pred_med = medfilt(y_pred, k)
        acc_med = np.sum(y_test == y_pred_med) / float(y_test.size)
        if max_acc_med < acc_med:
            max_acc_med = acc_med
            best_k = k
    plot(medfilt(y_pred, best_k))
    plot(y_test)
    print 'Raw acc: {:.3f}'.format(np.sum(y_test == y_pred) / float(y_test.size))
    print 'Smoothed acc (k = {}): {:.3f}'.format(best_k, max_acc_med)
    
def majority_voting(y_test, y_pred, loc):
    acc = 0
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)
    loc = loc.astype(int)
    for (i, idx) in enumerate(loc):    
        major = np.argmax(np.bincount(y_pred[idx[0]: idx[1]], minlength=10))
        print i/2, major
        acc += (i/2 == major)
    return acc

# <codecell>

y_pred_mfcc = spk_id(clf, X_train_mfcc, X_test_mfcc, y_train)

# <codecell>

majority_voting(y_test, y_pred_mfcc, loc_test)

# <codecell>

smooth(y_test, y_pred_mfcc, max_k=30)

# <codecell>

y_pred_sf = spk_id(clf, X_train_sf, X_test_sf, y_train)

# <codecell>

majority_voting(y_test, y_pred_sf, loc_test)

# <codecell>

smooth(y_test, y_pred_sf, max_k=30)

# <codecell>

X_train = np.vstack((X_train_mfcc, X_train_sf))
X_test = np.vstack((X_test_mfcc, X_test_sf))
y_pred_all = spk_id(clf, X_train, X_test, y_train)

# <codecell>

smooth(y_test, y_pred_all, max_k=30)

# <headingcell level=1>

# Generating RAW STFT for Feature Learning

# <codecell>

n_fft = 1024
hop_length = 512

X_train = None
y_train = None
X_test = None
y_test = None

len_train = []
len_test = []

for (i, spk_dir) in enumerate(files):
    for wav_dir in spk_dir[:N_train]:
        wav, sr = load_timit(wav_dir)
        stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        len_train.append(stft.shape[1])
        if X_train is None:
            X_train = stft
            y_train = i * np.ones((X_train.shape[1], ))
        else:
            X_train = np.hstack((X_train, stft))
            y_train = np.hstack((y_train, i * np.ones((stft.shape[1], ))))
            
    for wav_dir in spk_dir[N_train:]:
        wav, sr = load_timit(wav_dir)
        stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        len_test.append(stft.shape[1])
        if X_test is None:
            X_test = stft
            y_test = i * np.ones((X_test.shape[1], ))
        else:
            X_test = np.hstack((X_test, stft))
            y_test = np.hstack((y_test, i * np.ones((stft.shape[1], ))))

#sio.savemat('spkID_Train%s.mat' % N_train, {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test})
sio.savemat('bwe_spk_dep%s.mat' % N_train, {'X_train': X_train, 'y_train': y_train, 'len_train': len_train,
                                            'X_test': X_test, 'y_test': y_test, 'len_test': len_test})

# <codecell>

n_fft = 1024
hop_length = 512

X = None
y = None

for (i, spk_dir) in enumerate(files):
    for wav_dir in spk_dir:
        wav, sr = load_timit(wav_dir)
        if X is None:
            X = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            y = i * np.ones((X.shape[1], ))
        else:
            stft = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            X = np.hstack((X, stft))
            y = np.hstack((y, i * np.ones((stft.shape[1], ))))
            
sio.savemat('spkID_Full.mat', {'X': X, 'y': y})

# <codecell>


