# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, pickle

import numpy as np
import scipy.io as sio
from scikits.audiolab import Sndfile, Format
from matplotlib.pyplot import *

import librosa
import gamma_gvpl as vpl

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

n_fft = 1024
hop_length = 512
spk_dir = 'dr1/fcjf0/'
files = glob.glob(TIMIT_DIR + spk_dir + '*.wav')

N_train = 8
N_test = 2
np.random.seed(98765)
idx = np.random.permutation(10)

W_complex_train = None
for file_dir in files[:N_train]:
    wav, sr = load_timit(file_dir)
    if W_complex_train is None:
        W_complex_train = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    else:
        W_complex_train = np.hstack((W_complex_train, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))) 
W_complex_test = None
for file_dir in files[N_train:]:
    wav, sr = load_timit(file_dir)
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

threshold = 0.005
old_obj = -np.inf
L = 50
maxiter = 100
cold_start = False
batch = True

sfd = vpl.SF_Dict(np.abs(W_complex_train.T), L=L, seed=98765)
obj = []
for i in xrange(maxiter):
    sfd.vb_e(cold_start=cold_start, batch=batch, disp=0)
    if sfd.vb_m(disp=1):
        break
    obj.append(sfd.obj)
    improvement = (sfd.obj - old_obj) / abs(sfd.obj)
    print 'After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.4f}'.format(i, sfd.obj, old_obj, improvement)
    if improvement < threshold:
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

plot(meanA.ravel(), '-o')
pass

# <codecell>

for l in xrange(L):
    figure(l)
    plot(sfd.U[l])
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

sf_encoder.vb_e(cold_start = False, batch=True)
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
write_wav(w_rec, 'rec_spk_fit_L{}_F{}_H{}_{}.wav'.format(L, n_fft, hop_length, str_cold_start))
w_rec_org = librosa.istft(W_complex_test, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(w_rec_org, 'rec_spk_org.wav')

# <codecell>

save_object(sfd, 'dr1_fcjf0_L{}_F{}_H{}_{}_Seed98765'.format(L, n_fft, hop_length, str_cold_start))

