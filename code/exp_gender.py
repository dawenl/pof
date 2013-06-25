# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, pickle

import numpy as np
import scipy.io as sio
from scikits.audiolab import Sndfile, Format
from matplotlib.pyplot import *

import librosa
import vpl

# <codecell>

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

def learn_prior(W, L, maxiter=50, seed=None):
    sfd = dp.SF_Dict(W, L=L, seed=seed)
    obj = []
    for i in xrange(maxiter):
        print 'ITERATION: {}'.format(i)
        sfd.vb_e()
        if sfd.vb_m():
            break
        obj.append(sfd.obj)
    return (sfd.U, sfd.gamma, sfd.alpha, obj)

def encode(W, U, gamma, alpha, seed=None):
    L, _ = U.shape
    sfd = dp.SF_Dict(W, L=L, seed=seed)
    sfd.U, sfd.gamma, sfd.alpha = U, gamma, alpha
    sfd.vb_e()
    return sfd.EA
    
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

threshold = 0.01
old_obj = -np.inf
L = 50
maxiter = 100
cold_start = False
batch = False

sfd = vpl.SF_Dict(np.abs(W_complex_train.T), L=L, seed=98765)
obj = []
for i in xrange(maxiter):
    sfd.vb_e(cold_start=cold_start, batch=batch, disp=0)
    if sfd.vb_m(disp=1):
        break
    obj.append(sfd.obj)
    improvement = (sfd.obj - old_obj) / abs(sfd.obj)
    print 'After ITERATION: {}\tObjective improvement: {:.4f}'.format(i, improvement)
    if (sfd.obj - old_obj) / abs(sfd.obj) < threshold:
        break
    old_obj = sfd.obj

# <codecell>

plot(obj)
pass

# <codecell>

subplot(211)
specshow(sfd.U.T)
colorbar()
subplot(212)
specshow(sfd.EA.T)
colorbar()
tight_layout()
pass

# <codecell>

meanA = np.mean(sfd.EA, axis=0, keepdims=True)

tmpA = sfd.EA / meanA
tmpU = sfd.U * meanA.T

subplot(211)
specshow(tmpU.T)
colorbar()
title('U')
subplot(212)
specshow(tmpA.T)
colorbar()
title('A')
tight_layout()
pass

# <codecell>

figure()
plot(sfd.alpha, '-o')
figure()
plot(np.sqrt(1./sfd.gamma))
pass

# <codecell>

sf_encoder = vpl.SF_Dict(np.abs(W_complex_test.T), L=L, seed=98765)
sf_encoder.U, sf_encoder.gamma, sf_encoder.alpha = sfd.U, sfd.gamma, sfd.alpha

batch = True
sf_encoder.vb_e(cold_start = False, batch=batch, maxiter=100, atol=0.005)
A = sf_encoder.EA

# <codecell>

W_rec_amp = np.exp(np.dot(A, sfd.U)).T
W_rec = W_rec_amp * np.exp(1j * np.angle(W_complex_test))

# <codecell>

subplot(311)
specshow(logspec(np.abs(W_complex_test)))
title('Original')
colorbar()
subplot(312)
specshow(logspec(W_rec_amp))
title('Reconstruction')
colorbar()
subplot(313)
specshow(W_rec_amp - np.abs(W_complex_test))
title('Reconstruction Error')
colorbar()
tight_layout()
pass

# <codecell>

str_cold_start = 'cold' if cold_start else 'warm'
w_rec = librosa.istft(W_rec, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(w_rec, 'rec_gen{}_fit_L{}_F{}_H{}_{}.wav'.format(gender, L, n_fft, hop_length, str_cold_start))
w_rec_org = librosa.istft(W_complex_test, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(w_rec_org, 'rec_gen{}_org.wav'.format(gender))

# <codecell>


