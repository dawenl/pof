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

f_dirs_all = !ls -d "$TIMIT_DIR"dr[1-6]/f*
m_dirs_all = !ls -d "$TIMIT_DIR"dr[1-6]/m*

n_spk = 5
np.random.seed(98765)
f_dirs = np.random.permutation(f_dirs_all)[:n_spk]
m_dirs = np.random.permutation(m_dirs_all)[:n_spk]

f_files = [glob.glob(spk_dir + '/*.wav') for spk_dir in f_dirs]
m_files = [glob.glob(spk_dir + '/*.wav') for spk_dir in m_dirs]

# <codecell>

print f_dirs
print m_dirs

# <codecell>

N_train = 8

# <codecell>

n_fft = 1024
hop_length = 512

X_train = None
X_test = None
y_train = None
y_test = None

for (i, spk_dir) in enumerate(f_files):
    for wav_dir in spk_dir[:N_train]:
        wav, sr = load_timit(wav_dir)
        S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S)
        if X_train is None:
            X_train = librosa.feature.mfcc(librosa.logamplitude(S), d=13)
            y_train = i * np.ones((X_train.shape[1], ))
        else:
            mfcc = librosa.feature.mfcc(librosa.logamplitude(S), d=13)
            X_train = np.hstack((X_train, mfcc))
            y_train = np.hstack((y_train, i * np.ones((mfcc.shape[1], ))))
            
    for wav_dir in spk_dir[N_train:]:
        wav, sr = load_timit(wav_dir)
        S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S)
        if X_test is None:
            X_test = librosa.feature.mfcc(librosa.logamplitude(S), d=13)
            y_test = i * np.ones((X_test.shape[1], )) 
        else:
            mfcc = librosa.feature.mfcc(librosa.logamplitude(S), d=13)
            X_test = np.hstack((X_test, mfcc))
            y_test = np.hstack((y_test, i * np.ones((mfcc.shape[1], ))))

for (i, spk_dir) in enumerate(m_files):
    for wav_dir in spk_dir[:N_train]:
        wav, sr = load_timit(wav_dir)
        S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S)
        mfcc = librosa.feature.mfcc(librosa.logamplitude(S), d=13)
        X_train = np.hstack((X_train, mfcc))
        y_train = np.hstack((y_train, (i + n_spk) * np.ones((mfcc.shape[1], ))))
            
    for wav_dir in spk_dir[N_train:]:
        wav, sr = load_timit(wav_dir)
        S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S)
        
        mfcc = librosa.feature.mfcc(librosa.logamplitude(S), d=13)
        X_test = np.hstack((X_test, mfcc))
        y_test = np.hstack((y_test, (i + n_spk) * np.ones((mfcc.shape[1], ))))

# <codecell>

meanX = np.mean(X_train, axis=1, keepdims=True)
stdX = np.std(X_train, axis=1, keepdims=True)

X_train = (X_train - meanX) / stdX
X_test = (X_test - meanX) / stdX

# <codecell>

d = sio.loadmat('feat_sf_L50_TIMIT_spk20_spkID_N10.mat')
X_train = d['A_train']
X_test = d['A_test']
y_train = d['y_train'].ravel()
y_test = d['y_test'].ravel()

# <codecell>

def diff_feat(X):
    tmp = X.copy()
    L = X.shape[0]
    X = np.vstack((X, np.hstack((np.zeros((L, 1)), np.diff(tmp, n=1, axis=1)))))
    X = np.vstack((X, np.hstack((np.zeros((L, 2)), np.diff(tmp, n=2, axis=1)))))
    return X

X_train = diff_feat(X_train)
X_test = diff_feat(X_test)

print X_train.shape, X_test.shape

# <codecell>

fig(figsize=(12, 4))
subplot(121)
specshow(X_train)
colorbar()
subplot(122)
specshow(X_test)
colorbar()
pass

# <codecell>

clf = svm.LinearSVC()
clf.fit(X_train.T, y_train)

# <codecell>

for (i, spk_dir) in enumerate(f_files):
    bincount = np.zeros((2 * n_spk, ))
    for wav_dir in spk_dir[N_train:]:
        wav, sr = load_timit(wav_dir)
        S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S)
    
        mfcc = librosa.feature.mfcc(librosa.logamplitude(S), d=13)
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        mfcc = mfcc / np.std(mfcc, axis=1, keepdims=True)
        pred = clf.predict(mfcc.T)
        
        bincount = bincount + np.bincount(pred.astype(int64), minlength=2 * n_spk)
        #print np.bincount(pred.astype(int64))
        y = np.argmax(np.bincount(pred.astype(int64)))
        print y, i
    print bincount
        
for (i, spk_dir) in enumerate(m_files):
    bincount = np.zeros((2 * n_spk, ))
    for wav_dir in spk_dir[N_train:]:
        wav, sr = load_timit(wav_dir)
        S = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S)
    
        mfcc = librosa.feature.mfcc(librosa.logamplitude(S), d=13)
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)
        mfcc = mfcc / np.std(mfcc, axis=1, keepdims=True)
        pred = clf.predict(mfcc.T)
        
        bincount = bincount + np.bincount(pred.astype(int64), minlength=2 * n_spk)
        #print np.bincount(pred.astype(int64))
        y = np.argmax(np.bincount(pred.astype(int64)))
        print y, i + n_spk
    print bincount
        
for spk in xrange(2 * n_spk):
    X = X_test[:, y_test == spk]
    pred = clf.predict(X.T)
    print np.bincount(pred.astype(int64))
    y = np.argmax(np.bincount(pred.astype(int64)))
    print y, spk

# <codecell>

acc = np.sum(y_test == clf.predict(X_test.T)) / float(X_test.shape[1])
print acc

# <codecell>

## save raw STFT
n_fft = 1024
hop_length = 512

X_train = None
y_train = None
X_test = None
y_test = None

for (i, spk_dir) in enumerate(f_files):
    for wav_dir in spk_dir[:N_train]:
        wav, sr = load_timit(wav_dir)
        if X_train is None:
            X_train = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            y_train = i * np.ones((X_train.shape[1], ))
        else:
            stft = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            X_train = np.hstack((X_train, stft))
            y_train = np.hstack((y_train, i * np.ones((stft.shape[1], ))))
            
    for wav_dir in spk_dir[N_train:]:
        wav, sr = load_timit(wav_dir)
        if X_test is None:
            X_test = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            y_test = i * np.ones((X_test.shape[1], ))
        else:
            stft = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            X_test = np.hstack((X_test, stft))
            y_test = np.hstack((y_test, i * np.ones((stft.shape[1], ))))
            
for (i, spk_dir) in enumerate(m_files):
    for wav_dir in spk_dir[:N_train]:
        wav, sr = load_timit(wav_dir)
        if X_train is None:
            X_train = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            y_train = (i + n_spk) * np.ones((X_train.shape[1], ))
        else:
            stft = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            X_train = np.hstack((X_train, stft))
            y_train = np.hstack((y_train, (i + n_spk) * np.ones((stft.shape[1], ))))
            
    for wav_dir in spk_dir[N_train:]:
        wav, sr = load_timit(wav_dir)
        if X_test is None:
            X_test = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            y_test = (i + n_spk) * np.ones((X_test.shape[1], ))
        else:
            stft = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
            X_test = np.hstack((X_test, stft))
            y_test = np.hstack((y_test, (i + n_spk) * np.ones((stft.shape[1], ))))

# <codecell>

sio.savemat('spkID_N10.mat', {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test})

# <codecell>


