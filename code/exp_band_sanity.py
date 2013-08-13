# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, pickle, time

import numpy as np
import scipy.io as sio
import scipy.stats as stats
from scikits.audiolab import Sndfile, Format
from matplotlib.pyplot import *

import librosa
import gamma_gvpl as vpl
import sf_gap_nmf as sf_nmf

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

X_complex_train = None
for file_dir in files[:N_train]:
    for wav_dir in file_dir:
        wav, sr = load_timit(wav_dir)
        if X_complex_train is None:
            X_complex_train = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        else:
            X_complex_train = np.hstack((X_complex_train, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))) 
X_complex_test = None
for file_dir in files[N_train:]:
    for wav_dir in file_dir:
        wav, sr = load_timit(wav_dir)
        if X_complex_test is None:
            X_complex_test = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        else:
            X_complex_test = np.hstack((X_complex_test, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))

# <codecell>

# load the prior learned from training data
d = sio.loadmat('priors/gamma_gender_batch.mat')
U = d['U']
gamma = d['gamma'].ravel()
alpha = d['alpha'].ravel()
L = alpha.size

# <headingcell level=1>

# Sanity check 1: use the prior to fit the original training data, but bandlimited

# <codecell>

# cut-off above 3000Hz
freq_threshold = 3000.
bin_cutoff = n_fft * freq_threshold / sr
X_cutoff_train = X_complex_train[:(bin_cutoff+1)]

x_cutoff_train = librosa.istft(X_cutoff_train, n_fft=2*bin_cutoff, hop_length=bin_cutoff, hann_w=0)
write_wav(x_cutoff_train, 'prior_be_cutoff.wav', samplerate=2 * freq_threshold)

# <codecell>

reload(vpl)
encoder_train = vpl.SF_Dict(np.abs(X_cutoff_train.T), L=L, seed=98765)
encoder_train.U, encoder_train.gamma, encoder_train.alpha = U[:, :(bin_cutoff+1)], gamma[:(bin_cutoff+1)], alpha

encoder_train.vb_e(cold_start = False)
A = encoder_train.EA

# <codecell>

EX_train = np.zeros_like(np.abs(X_complex_train.T))
for t in xrange(EX_train.shape[0]):
    EX_train[t] = np.exp(np.sum(vpl.comp_log_exp(encoder_train.a[t, :, np.newaxis], encoder_train.b[t, :, np.newaxis], U), axis=0))

# <codecell>

fig()
subplot(121)
specshow(logspec(EX_train.T))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
subplot(122)
specshow(logspec(np.abs(X_complex_train)))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
pass

# <codecell>

## mean of predictive log-likelihood
pred_likeli = np.mean(stats.expon.logpdf(np.abs(X_complex_train[(bin_cutoff+1):]), scale=EX_train[(bin_cutoff+1):]))
print pred_likeli

# <codecell>

write_wav(librosa.istft(EX_train.T * (X_complex_train / np.abs(X_complex_train)), n_fft=n_fft, hann_w=0, hop_length=hop_length), 'be_sanity_check1.wav')

# <headingcell level=1>

# Sanity check 2: use the prior to fit the band-limited held-out data

# <codecell>

X_cutoff_test = X_complex_test[:(bin_cutoff+1)]

x_cutoff_test = librosa.istft(X_cutoff_test, n_fft=2*bin_cutoff, hop_length=bin_cutoff, hann_w=0)
write_wav(x_cutoff_test, 'prior_be_cutoff.wav', samplerate=2 * freq_threshold)

# <codecell>

reload(vpl)
encoder_test = vpl.SF_Dict(np.abs(X_cutoff_test.T), L=L, seed=98765)
encoder_test.U, encoder_test.gamma, encoder_test.alpha = U[:, :(bin_cutoff+1)], gamma[:(bin_cutoff+1)], alpha

encoder_test.vb_e(cold_start = False)
A = encoder_test.EA

# <codecell>

EX_test = np.zeros_like(np.abs(X_complex_test))
for t in xrange(EX_test.shape[0]):
    EX_test[t] = np.exp(np.sum(vpl.comp_log_exp(encoder_test.a[t, :, np.newaxis], encoder_test.b[t, :, np.newaxis], U), axis=0))

# <codecell>

fig()
subplot(121)
specshow(logspec(EX_test.T))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
subplot(122)
specshow(logspec(np.abs(X_complex_test)))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
pass

# <codecell>

## mean of predictive log-likelihood
pred_likeli = np.mean(stats.expon.logpdf(np.abs(X_complex_test[(bin_cutoff+1):]), scale=EX_test[(bin_cutoff+1):]))
print pred_likeli

# <codecell>

write_wav(librosa.istft(EX_test.T * (X_complex_test / np.abs(X_complex_test)), n_fft=n_fft, hann_w=0, hop_length=hop_length), 'be_sanity_check2.wav')

# <headingcell level=1>

# Sanity check 3: use the prior to fit the original band-limied training data under the full NMF setting

# <codecell>

reload(sf_nmf)
U = d['U'].T
sfnmf = sf_nmf.SF_GaP_NMF(np.abs(X_cutoff_train), U[:(bin_cutoff+1)], gamma[:(bin_cutoff+1)], alpha, K=100, seed=98765)

# <codecell>

score = sfnmf.bound()
criterion = 0.0005
objs = []
for i in xrange(1000):
    start_t = time.time()
    sfnmf.update(disp=1)
    t = time.time() - start_t
    
    lastscore = score
    score = sfnmf.bound()
    objs.append(score)
    improvement = (score - lastscore) / abs(lastscore)
    print ('iteration {}: bound = {:.2f} ({:.5f} improvement) time = {:.2f}'.format(i, score, improvement, t))
    if improvement < criterion:
        break

# <codecell>

plot(objs)
pass

# <codecell>

sfnmf.figures()

# <codecell>

goodk = sfnmf.goodk()
fig()
subplot(121)
specshow(sfnmf.Ea[:, goodk])
title('A')
colorbar()
subplot(122)
specshow(logspec(sfnmf.Ew[:, goodk]))
title('W')
colorbar()
tight_layout()
pass

# <codecell>

## Infer the high-frequency contents
#Ew = np.zeros((U.shape[0], goodk.size))
#for (i, k) in enumerate(goodk[sfnmf.Et[goodk] > 1e-7]):
#    Ew[:, i] = np.exp(np.sum(vpl.comp_log_exp(sfnmf.nua[:, k], sfnmf.rhoa[:, k], -U), axis=1))
Ew = np.exp(np.dot(U, sfnmf.Ea[:, goodk]))

# <codecell>

c = np.mean(sfnmf.X / sfnmf._xtwid(goodk))
X_bar = c * np.dot(Ew * sfnmf.Et[goodk], sfnmf.Eh[goodk])

fig()
subplot(121)
specshow(logspec(X_bar))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
subplot(122)
specshow(logspec(np.abs(X_complex_train)))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
pass

# <codecell>

## mean of predictive log-likelihood
pred_likeli = np.mean(stats.expon.logpdf(np.abs(X_complex_train[(bin_cutoff+1):]), scale=X_bar[(bin_cutoff+1):]))
print pred_likeli

# <codecell>

write_wav(librosa.istft(X_bar * (X_complex_train / np.abs(X_complex_train)), n_fft=n_fft, hann_w=0, hop_length=hop_length), 'be_sanity_check3.wav')

# <codecell>


