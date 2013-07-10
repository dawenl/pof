# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, pickle

import numpy as np
import scipy.io as sio
from scikits.audiolab import Sndfile, Format
from matplotlib.pyplot import *

import librosa
import st_vpl 

# <codecell>

fig = functools.partial(figure, figsize=(16,4))
specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

def logspec(X, amin=1e-10, dbdown=80):
    logX = 20 * np.log10(np.maximum(X, amin))
    return np.maximum(logX, logX.max() - dbdown)

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    pass

def load_object(filename):
    with open(filename, 'r') as output:
        obj = pickle.load(output)
    return obj 

# <codecell>

TIMIT_DIR = '../../timit/train/'

# <codecell>

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

gender = 'f'
dirs = !ls -d "$TIMIT_DIR"dr1/"$gender"*

files = [glob.glob(spk_dir + '/*.wav') for spk_dir in dirs]
n_files = len(files)

# <codecell>

n_fft = 1024
hop_length = 512

N_train = int(0.8 * n_files)
N_test = n_files - N_train
np.random.seed(98765)

idx = np.random.permutation(n_files)

W_complex_train = None
for file_dir in files[:N_train]:
    for wav_dir in file_dir:
        wav, sr = load_timit(wav_dir)
        if W_complex_train is None:
            W_complex_train = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        else:
            W_complex_train = np.hstack((W_complex_train, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))) 
W_complex_test = None
for file_dir in files[N_train:]:
    for wav_dir in file_dir:
        wav, sr = load_timit(wav_dir)
        if W_complex_test is None:
            W_complex_test = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        else:
            W_complex_test = np.hstack((W_complex_test, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))

# <codecell>

subplot(211)
specshow(logspec(np.abs(W_complex_train)))
colorbar() 
subplot(212)
specshow(logspec(np.abs(W_complex_test)))
colorbar()
pass

# <codecell>

TAU = 1.
KAPPA = .75
rho = (arange(100) + TAU)**(-KAPPA)
plot(rho, '-o')

# <codecell>

reload(st_vpl)
threshold = 0.01
old_obj = -np.inf
L = 50
maxiter = 500
batch = True

obj = []

TAU = 1.
KAPPA = 0.75
batch_size = 100

sfd = st_vpl.SF_Dict(np.ones((batch_size, W_complex_train.shape[0])), L=L, seed=98765)
n_total = W_complex_train.shape[1]
for i in xrange(maxiter):
    rho = 0.5 * (i + TAU)**(-KAPPA)
    idx = np.random.choice(n_total, size=batch_size, replace=False)
    sfd.switch(np.abs(W_complex_train[:,idx].reshape(batch_size,-1)))
    sfd.vb_e(cold_start=True, batch=batch, disp=0)
    if sfd.vb_m(rho, batch=False, disp=1):
        break
    obj.append(sfd.obj)
    improvement = (sfd.obj - old_obj) / abs(sfd.obj)
    print 'After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.4f}'.format(i, sfd.obj, old_obj, improvement)
    old_obj = sfd.obj

# <codecell>

plot(obj)
pass

# <codecell>

specshow(sfd.U.T)
colorbar()
pass

# <codecell>

for l in xrange(L):
    fig()
    subplot(121)
    plot(sfd.U[l])
    subplot(122)
    plot(np.exp(sfd.U[l]))
    tight_layout()
pass

# <codecell>

fig()
subplot(121)
plot(flipud(sort(sfd.alpha)), '-o')
subplot(122)
plot(np.sqrt(1./sfd.gamma))
pass

# <codecell>

W_rec = np.exp(np.dot(sfd.EA, sfd.U)).T
subplot(311)
specshow(logspec(W_rec))
colorbar()
subplot(312)
specshow(logspec(np.abs(W_complex_train[:,idx])))
colorbar()
subplot(313)
specshow(W_rec - np.abs(W_complex_train[:,idx]))
colorbar()
pass

# <codecell>


