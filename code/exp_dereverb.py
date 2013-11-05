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

lengths = [0]
x = None
x_rev = None
X_complex = None
X_rev_complex = None

for spk_dir in files:
    for wav_dir in spk_dir:
        wav, sr = load_timit(wav_dir)
        wav_rev = np.convolve(wav, h)[:wav.size]
        stft = librosa.stft(wav, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        stft_rev = librosa.stft(wav_rev, n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
        lengths.append(stft.shape[1])
        if X_complex is None:
            X_complex = stft
            X_rev_complex = stft_rev
            x = wav
            x_rev = wav_rev
        else:
            X_complex = np.hstack((X_complex, stft))
            X_rev_complex = np.hstack((X_rev_complex, stft_rev))
            x = np.hstack((x, wav))
            x_rev = np.hstack((x_rev, wav_rev))
X = np.abs(X_complex)
X_rev = np.abs(X_rev_complex)
print X.shape, X_rev.shape

write_wav(x, 'x_org.wav')
write_wav(x_rev, 'x_rev.wav')

# <codecell>

fig(figsize=(8, 8))
subplot(211)
specshow(logspec(X))
colorbar()
subplot(212)
specshow(logspec(X_rev))
colorbar()
pass

# <codecell>

# load the prior learned from training data
prior_mat = sio.loadmat('priors/sf_L30_TIMIT_spk20.mat')
U = prior_mat['U']
gamma = prior_mat['gamma'].ravel()
alpha = prior_mat['alpha'].ravel()
L = alpha.size

# <codecell>

reload(vpl)
encoder = vpl.SF_Dict(X_rev.T, U, alpha, gamma, L=L, seed=98765, flat_init=False)

# <codecell>

threshold = 0.0001
old_obj = -np.inf
maxiter = 200
obj = []
for i in xrange(maxiter):
    encoder.vb_m()
    encoder.vb_e(cold_start=False, disp=1)
    score = encoder.bound()
    obj.append(score)
    improvement = (score - old_obj) / abs(old_obj)
    print 'After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.5f}'.format(i, score, old_obj, improvement)
    if improvement < threshold and i >= 5:
        break
    old_obj = score

# <codecell>

plot(obj)

# <codecell>

EX_sub = X_rev / np.exp(encoder.reverb[:, np.newaxis])

# <codecell>

fig(figsize=(16, 8))
subplot(221)
specshow(logspec(X), vmin=-75, vmax=25)
colorbar()
subplot(222)
specshow(logspec(X_rev), vmin=-75, vmax=25)
colorbar()
subplot(223)
specshow(logspec(EX_sub), vmin=-75, vmax=25)
colorbar()
pass

# <codecell>

x_dr_np = librosa.istft(X_rev_complex * (EX_sub / X_rev), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
write_wav(x_dr_np, 'x_dr_np.wav')

x_dr_op = librosa.istft(X_complex * (EX_sub / X), n_fft=n_fft, hann_w=hann_w, hop_length=hop_length)
write_wav(x_dr_op, 'x_dr_op.wav')

# <codecell>

fig()
subplot(121)
logX_diff = np.log(X_rev) - np.log(X)
specshow(logX_diff, vmin=np.amin(logX_diff), vmax=np.amax(logX_diff))
colorbar()
subplot(122)
logX_drev_diff = np.log(EX_sub) - np.log(X)
specshow(logX_drev_diff, vmin=np.amin(logX_diff), vmax=np.amax(logX_diff))
colorbar()
pass

# <codecell>

bins = np.linspace(0, np.amax(logX_diff), num=50)
hist(np.abs(logX_diff).ravel(), bins=bins, alpha=0.5)
hist(np.abs(logX_drev_diff).ravel(), bins=bins, alpha=0.5)
legend(['reverb', 'derevb'])
pass

# <codecell>

fig(figsize=(16, 8))
subplot(211)
plot(logX_diff.mean(axis=1))
plot(np.log(H))
plot(encoder.reverb)
axhline(y=0, color='k')
subplot(212)
plot(logX_diff.mean(axis=1) - encoder.reverb)
axhline(y=0, color='r')
ylim([-5, 2])
pass

# <codecell>


