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

rir_type = 'lecture'
rir_param = '1_1_5'

rir_mat = sio.loadmat('air_binaural_%s_%s.mat' % (rir_type, rir_param))
h = rir_mat['h_air'].ravel()
h = samplerate.resample(h, 1./3, 'sinc_best')
plot(h)
H = np.abs(np.fft.fft(h, n_fft)[:n_fft/2+1])
fig()
plot(20 * np.log10(H))
fig()
plot(20 * np.log10(np.abs(np.fft.fft(h)))[:h.size/2+1])
pass

# <codecell>

# load the prior learned from training data
prior_mat = sio.loadmat('priors/sf_L30_TIMIT_spk20.mat')
U = prior_mat['U']
gamma = prior_mat['gamma'].ravel()
alpha = prior_mat['alpha'].ravel()
L = alpha.size

# <codecell>

def upsample_filters(U, gamma):
    L, F = U.shape
    full_U = np.hstack((U, np.fliplr(U[:, 1:-1])))
    U_pad = np.zeros((L, 2 * F - 1))
    for l in xrange(L):
        tmp = samplerate.resample(full_U[l], 2, 'sinc_best')
        U_pad[l] = tmp[: 2 * F - 1]
    gamma_pad = samplerate.resample(gamma, 2, 'sinc_best')
    return (U_pad, gamma_pad)

U, gamma = upsample_filters(U, gamma)
print U.shape, gamma.shape

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

spk_color = sio.loadmat('spk_color_global.mat')['spk_color']

# <codecell>

fig()
plot(LOG_TO_DB * (reverbs_me.T - spk_color).mean(axis=1))
plot(LOG_TO_DB * reverbs_em.mean(axis=0))
plot(20 * np.log10(H))
legend(['ME', 'EM', 'ground truth'])
pass

# <codecell>

reload(vpl)

for (i, spk_dir) in enumerate(files, 1):
    for (j, wav_dir) in enumerate(spk_dir, 1):
        flat_init = True
        
        wav, sr = load_timit(wav_dir)
        wav_rev = np.convolve(wav, h)[:wav.size]
        X = librosa.stft(wav, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        X_rev = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        
        x = librosa.istft(X, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        x_rev = librosa.istft(X_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x, 'reverb_%s/spk%s_sent%s_org.wav' % (rir_type, i, j))
        write_wav(x_rev, 'reverb_%s/spk%s_sent%s_rev.wav' % (rir_type, i, j))

        EX_em = np.abs(X_rev) / np.exp(reverbs_em[i-1, :, np.newaxis])
        EX_me = np.abs(X_rev) / np.exp(reverbs_me[i-1, :, np.newaxis] - spk_color)
        
        x_dr_em_np = librosa.istft(X_rev * (EX_em / np.abs(X_rev)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_em_np, 'reverb_%s/spk%s_sent%s_dr_np_em.wav' % (rir_type, i, j))
        x_dr_me_np = librosa.istft(X_rev * (EX_me / np.abs(X_rev)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_me_np, 'reverb_%s/spk%s_sent%s_dr_np_me.wav' % (rir_type, i, j))

        x_dr_em_op = librosa.istft(X * (EX_em / np.abs(X)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_em_op, 'reverb_%s/spk%s_sent%s_dr_op_em.wav' % (rir_type, i, j))
        x_dr_me_op = librosa.istft(X * (EX_me / np.abs(X)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_me_op, 'reverb_%s/spk%s_sent%s_dr_op_me.wav' % (rir_type, i, j))
pass

# <codecell>

for (i, spk_dir) in enumerate(files, 1):
    for (j, wav_dir) in enumerate(spk_dir, 1):
        wav, sr = load_timit(wav_dir)
        wav_rev = np.convolve(wav, h)[:wav.size]
        X = librosa.stft(wav, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        X_rev = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        
        logX_diff = np.log(np.abs(X_rev)) - np.log(np.abs(X))   
        EX_emp = np.abs(X_rev) / np.exp(np.mean(logX_diff, axis=1, keepdims=True))
        #EX_emp = np.abs(X_rev) / H[:, np.newaxis]
        
        x_dr_emp = librosa.istft(X_rev * (EX_emp / np.abs(X_rev)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_emp, 'reverb_%s/spk%s_sent%s_dr_emp.wav' % (rir_type, i, j))
pass

# <codecell>

X_train = sio.loadmat('TIMIT_spk20.mat')['W']
mean_spk = X_train.mean(axis=1, keepdims=True)

# <codecell>

plot(20 * log10(mean_spk))

# <codecell>

for (i, spk_dir) in enumerate(files, 1):
    for (j, wav_dir) in enumerate(spk_dir, 1):
        wav, sr = load_timit(wav_dir)
        wav_rev = np.convolve(wav, h)[:wav.size]
        X_rev = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        
        #mean_spk = np.mean(sio.loadmat('spk_dep_dr/spk%s.mat' % i)['W'], axis=1, keepdims=True)
        EX_cmn = np.abs(X_rev) / np.mean(np.abs(X_rev), axis=1, keepdims=True) * mean_spk
        
        fig()
        plot(20 * log10(np.mean(np.abs(X_rev), axis=1, keepdims=True) / mean_spk))
        plot(20 * log10(H))
        
        x_dr_cmn = librosa.istft(X_rev * (EX_cmn / np.abs(X_rev)), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        write_wav(x_dr_cmn, 'reverb_%s/spk%s_sent%s_dr_cmn.wav' % (rir_type, i, j))
pass

# <codecell>


