# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, pickle, time

import numpy as np
import scipy.io as sio
from scikits.audiolab import Sndfile, Format

import librosa
import reverb_gvpl as vpl

import scikits.samplerate as samplerate

# <codecell>

fig = functools.partial(figure, figsize=(16,4))
specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

LOG_TO_DB = 20 * log10(e)

def logspec(X, amin=1e-10, dbdown=100):
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

n_fft = 1024
hann_w = 1024
hop_length = 256

# <codecell>

# load the prior learned from training data
prior_mat = sio.loadmat('priors/sf_L30_TIMIT_spk20.mat')
U = prior_mat['U']
gamma = prior_mat['gamma'].ravel()
alpha = prior_mat['alpha'].ravel()
L = alpha.size

# <codecell>

def learn_reverb(encoder, threshold=0.0001, maxiter=200, flat_init=True):
    old_obj = -np.inf
    for i in xrange(maxiter):
        if flat_init:
            encoder.vb_e(cold_start=False, disp=1)
            encoder.vb_m()  
        else:
            encoder.vb_m()
            encoder.vb_e(cold_start=False, disp=1)
        score = encoder.bound()
        improvement = (score - old_obj) / abs(old_obj)
        print('After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.5f}'.format(i, score, 
                                                                                                          old_obj, improvement))
        if improvement < threshold and i >= 5:
            break
        old_obj = score
    pass

# <codecell>

## load pre-trained reverbs (if any)
tmp = sio.loadmat('rir_%s_me.mat' % rir_type)
reverbs_me = tmp['rir']

tmp = sio.loadmat('rir_%s_em.mat' % rir_type)
reverbs_em = tmp['rir']

# <codecell>

fig()
plot(LOG_TO_DB * reverbs_me.mean(axis=0))
plot(LOG_TO_DB * reverbs_em.mean(axis=0))
plot(20 * np.log10(H))
legend(['ME', 'EM', 'ground truth'])
pass

# <codecell>

reload(vpl)

n_spk = 6

reverbs_me = sio.loadmat('rir_iPhone_me.mat')['rir']
spk_color = sio.loadmat('spk_color_global_1.mat')['spk_color']

for i in xrange(n_spk):
    wav_rev, sr = librosa.load('iPhone/spk%s_sent1_rev.wav' % (i+1), sr=None)
    X_rev_complex = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    X_rev = np.abs(X_rev_complex)

    EX_me = X_rev / np.exp(reverbs_me[i, :, np.newaxis] - spk_color)
        
    x_dr_me_np = librosa.istft(X_rev_complex * (EX_me / X_rev), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    write_wav(x_dr_me_np, 'iPhone/spk%s_sent1_dr_np_me.wav' % (i+1))
pass

# <codecell>

fig()
plot(spk_color)
pass

# <codecell>

sio.savemat('rir_iPhone_em.mat', {'rir': reverbs_em})

# <codecell>

for i in xrange(n_spk):
    wav, sr = librosa.load('iPhone/spk%s_sent1_org.wav' % (i+1), sr=None)
    wav_rev, _ = librosa.load('iPhone/spk%s_sent1_rev.wav' % (i+1), sr=None)
    X = librosa.stft(wav, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    X_rev = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        
    logX_diff = np.log(np.abs(X_rev)) - np.log(np.abs(X))   
    EX_emp = np.abs(X_rev) / np.exp(np.mean(logX_diff, axis=1, keepdims=True))
        
    x_dr_emp = librosa.istft(X_rev * (EX_emp / np.abs(X_rev)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    write_wav(x_dr_emp, 'iPhone/spk%s_sent1_dr_emp.wav' % (i+1))
pass

# <codecell>

W = sio.loadmat('TIMIT_spk20.mat')['W']
mean_spk = W.mean(axis=1, keepdims=True)

# <codecell>

fig()
plot(20 * log10(mean_spk))

# <codecell>

for i in xrange(n_spk):
    wav_rev, sr = librosa.load('iPhone/spk%s_sent1_rev.wav' % (i+1), sr=None)
    X_rev_complex = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    X_rev = np.abs(X_rev_complex)
        
    EX_cmn = X_rev / np.mean(X_rev, axis=1, keepdims=True) * mean_spk
        
    x_dr_cmn = librosa.istft(X_rev_complex * (EX_cmn / X_rev), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    write_wav(x_dr_cmn, 'iPhone/spk%s_sent1_dr_cmn.wav' % (i+1))
pass

# <codecell>


