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
#X_complex_test = None
#for file_dir in files[N_train:]:
#    for wav_dir in file_dir:
#        wav, sr = load_timit(wav_dir)
#        if X_complex_test is None:
#            X_complex_test = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
#        else:
#            X_complex_test = np.hstack((X_complex_test, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))

# <headingcell level=1>

# Sanity check 1: use the prior to fit the original training data, but bandlimited

# <codecell>

# cut-off above 3000Hz
freq_threshold = 3000.
bin_cutoff = n_fft * freq_threshold / sr
X_cutoff = X_complex_train[:(bin_cutoff+1)]

x_cutoff = librosa.istft(X_cutoff, n_fft=2*bin_cutoff, hop_length=bin_cutoff, hann_w=0)
write_wav(x_cutoff, 'prior_be_cutoff.wav', samplerate=2 * freq_threshold)

# <codecell>

d = sio.loadmat('priors/gamma_gender_batch.mat')
U = d['U']
gamma = d['gamma'].ravel()
alpha = d['alpha'].ravel()
L = alpha.size

# <codecell>

reload(vpl)
sf_encoder = vpl.SF_Dict(np.abs(X_cutoff.T), L=L, seed=98765)
sf_encoder.U, sf_encoder.gamma, sf_encoder.alpha = U[:, :(bin_cutoff+1)], gamma[:(bin_cutoff+1)], alpha

sf_encoder.vb_e(cold_start = False)
A = sf_encoder.EA

# <codecell>

EX = np.zeros_like(np.abs(X_complex_train))
for t in xrange(EX.shape[0]):
    EX[t] = np.exp(np.sum(vpl.comp_log_exp(sf_encoder.a[t], sf_encoder.b[t], U), axis=0))

