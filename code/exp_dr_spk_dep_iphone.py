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

reload(vpl)

n_spk = 6
reverbs_em = np.zeros((n_spk, n_fft/2+1))
        
for i in xrange(n_spk):
    wav_rev, sr = librosa.load('iPhone/spk%s_sent1_rev.wav' % (i+1), sr=None)
    X_rev_complex = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    X_rev = np.abs(X_rev_complex)
    
    prior_mat = sio.loadmat('spk_dep_dr/sf_L5_spk%s.mat' % (i+1))
    print 'Loading prior for spk%s...' % (i+1)
    U = prior_mat['U']
    gamma = prior_mat['gamma'].ravel()
    alpha = prior_mat['alpha'].ravel()
    L = alpha.size
    
    encoder = vpl.SF_Dict(X_rev.T, U, alpha, gamma, L=L, seed=98765)
    print 'Learning spk%s for sent1' % (i+1)
    learn_reverb(encoder)
    reverbs_em[i] = encoder.reverb.copy()

    EX_em = X_rev / np.exp(reverbs_em[i, :, np.newaxis])
        
    x_dr_em_np = librosa.istft(X_rev_complex * (EX_em / X_rev), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    write_wav(x_dr_em_np, 'iPhone_dep/spk%s_sent1_dr_np_em.wav' % (i+1))
pass

# <codecell>

sio.savemat('rir_iPhone_em_spk_dep_L%s.mat' % L, {'rir':reverbs_em})

# <codecell>

fig()
plot(LOG_TO_DB * reverbs_em.mean(axis=0))
pass

# <codecell>

for (i, spk_dir) in enumerate(files, 1):
    for (j, wav_dir) in enumerate(spk_dir, 1):
        wav, sr = load_timit(wav_dir)
        wav_rev = np.convolve(wav, h)[:wav.size]
        X = librosa.stft(wav, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        X_rev = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        
        logX_diff = np.log(np.abs(X_rev)) - np.log(np.abs(X))       
        EX_emp = X_rev / np.exp(np.mean(logX_diff, axis=1, keepdims=True))
        
        x_dr_emp = librosa.istft(X_rev * (EX_emp / np.abs(X_rev)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_emp, 'reverb_%s/spk%s_sent%s_dr_emp.wav' % (rir_type, i, j))
pass

# <codecell>

# load the prior learned from training data
prior_mat = sio.loadmat('priors/sf_L30_TIMIT_spk20.mat')
U = prior_mat['U']
gamma = prior_mat['gamma'].ravel()
alpha = prior_mat['alpha'].ravel()
L = alpha.size

# <codecell>

spk_color = np.zeros((n_spk, n_fft/2+1))

for i in xrange(n_spk):
    train_data = sio.loadmat('spk_dep_dr/spk%s.mat' % (i+1))
    W = train_data['W']
            
    print 'Learning coloration for spk%s' % (i+1)
    encoder = vpl.SF_Dict(W.T, U, alpha, gamma, L=L, seed=98765, flat_init=False)
    learn_reverb(encoder, flat_init=False)
    spk_color[i] = encoder.reverb.copy()

# <codecell>

fig()
plot(spk_color.T)

# <codecell>

sio.savemat('spk_color.mat', {'spk_color':spk_color})
fig()
plot(spk_color.T)

# <codecell>

reload(vpl)

reverbs_me = sio.loadmat('rir_iPhone_me_spk_dep.mat')['rir']

for i in xrange(n_spk):
    wav_rev, sr = librosa.load('iPhone/spk%s_sent1_rev.wav' % (i+1), sr=None)
    X_rev_complex = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    X_rev = np.abs(X_rev_complex)
    
    reverb = reverbs_me[i] - spk_color[i]
    
    EX_me = X_rev / np.exp(reverb[:, np.newaxis])
        
    x_dr_me_np = librosa.istft(X_rev_complex * (EX_me / X_rev), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
    write_wav(x_dr_me_np, 'iPhone_dep/spk%s_sent1_dr_np_me.wav' % (i+1))
pass

# <codecell>


