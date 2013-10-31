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

TIMIT_DIR = '../../timit/test/'

# <codecell>

f_dirs_all = !ls -d "$TIMIT_DIR"dr[1-3]/f*
m_dirs_all = !ls -d "$TIMIT_DIR"dr[1-5]/m*

n_spk = 3
np.random.seed(98765)
f_dirs = np.random.permutation(f_dirs_all)[:n_spk]
m_dirs = np.random.permutation(m_dirs_all)[:n_spk]

files = [glob.glob(spk_dir + '/*.wav')[:1] for spk_dir in f_dirs]
files.extend([glob.glob(spk_dir + '/*.wav')[:1] for spk_dir in m_dirs])

# <codecell>

files

# <codecell>

n_fft = 1024
hann_w = 1024
hop_length = 256

# <codecell>

rir_type = 'meeting'
rir_param = '0_1_1'

rir_mat = sio.loadmat('air_binaural_%s_%s.mat' % (rir_type, rir_param))
h = rir_mat['h_air'].ravel()
h = samplerate.resample(h, 1./3, 'sinc_best')
plot(h)
H = np.abs(np.fft.fft(h, n_fft)[:n_fft/2+1])
fig()
plot(20 * np.log10(H))
pass

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

#reverbs_em = np.zeros((len(files), n_fft/2+1))
reverbs_me = np.zeros((len(files), n_fft/2+1))

for (i, spk_dir) in enumerate(files, 1):
    for (j, wav_dir) in enumerate(spk_dir, 1):
        wav, sr = load_timit(wav_dir)
        wav_rev = np.convolve(wav, h)[:wav.size]
        X = librosa.stft(wav, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        X_rev = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        
        prior_mat = sio.loadmat('spk_dep_dr/sf_L30_spk%s.mat' % i)
        print 'Loading prior for spk%s...' % i
        U = prior_mat['U']
        gamma = prior_mat['gamma'].ravel()
        alpha = prior_mat['alpha'].ravel()
        L = alpha.size
        
        encoder = vpl.SF_Dict(np.abs(X_rev.T), U, alpha, gamma, L=L, seed=98765, flat_init=False)
        print 'Learning spk%s for sent%s' % (i, j)
        learn_reverb(encoder, flat_init=False)
            
        EX_me = np.abs(X_rev) / np.exp(encoder.reverb[:, np.newaxis])
        reverbs_me[i-1] = encoder.reverb.copy()

        #x_dr_em_np = librosa.istft(X_rev * (EX_em / np.abs(X_rev)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        #write_wav(x_dr_em_np, 'spk_dep_reverb_%s/spk%s_sent%s_dr_np_em.wav' % (rir_type, i, j))
        #x_dr_em_op = librosa.istft(X * (EX_em / np.abs(X)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        #write_wav(x_dr_em_op, 'spk_dep_reverb_%s/spk%s_sent%s_dr_op_em.wav' % (rir_type, i, j))
        
        x_dr_me_np = librosa.istft(X_rev * (EX_me / np.abs(X_rev)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_me_np, 'spk_dep_reverb_%s/spk%s_sent%s_dr_np_me.wav' % (rir_type, i, j))
        x_dr_me_op = librosa.istft(X * (EX_me / np.abs(X)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_me_op, 'spk_dep_reverb_%s/spk%s_sent%s_dr_op_me.wav' % (rir_type, i, j))
pass

# <codecell>

#sio.savemat('rir_%s_em_spk_dep.mat' % rir_type, {'rir':reverbs_em})
sio.savemat('rir_%s_me_spk_dep.mat' % rir_type, {'rir':reverbs_me})

# <codecell>

fig()
plot(LOG_TO_DB * reverbs.mean(axis=0))
plot(20 * np.log10(H))
legend(['EM', 'ground truth'])
pass

# <codecell>

rir_mat = sio.loadmat('rir_meeting.mat')
reverbs = rir_mat['rir']

# <codecell>

rir_mat = sio.loadmat('air_binaural_meeting_0_1_1.mat')
h = rir_mat['h_air'].ravel()
h = samplerate.resample(h, 1./3, 'sinc_best')
plot(h)
H = np.abs(np.fft.fft(h, n_fft)[:n_fft/2+1])
fig()
plot(20 * np.log10(H))
pass

# <codecell>

fig()
plot(LOG_TO_DB * reverbs[0])
plot(20 * np.log10(H))
plot(LOG_TO_DB * (reverbs[0] - spk_color.mean(axis=0)))
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

fig()
plot(reverbs.mean(axis=0))
plot(np.log(H))

# <codecell>

def ceps_dist(X, Y, order=24, cmn=True):
    X_sym = np.vstack((X, X[1:-1]))
    Y_sym = np.vstack((Y, Y[1:-1]))
    ceps_X = realceps(X_sym)[:order+1]
    ceps_Y = realceps(Y_sym)[:order+1]
    if cmn:
        ceps_X = ceps_X - np.mean(ceps_X, axis=1, keepdims=True)
        ceps_Y = ceps_Y = np.mean(ceps_Y, axis=1, keepdims=True)
    err = (ceps_X - ceps_Y)**2
    #return err
    ds = 10 / log(10) * np.sqrt(2 * np.sum(err[1:], axis=0) + err[0, :])
    return np.mean(ds), np.median(ds)

def realceps(X, flr=-100):
    flr = np.amax(X) * 10**(flr/20)
    X = np.maximum(X, flr)
    return np.real(np.fft.ifft(np.log(X), axis=0))

