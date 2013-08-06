# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, pickle

import numpy as np
import scipy.io as sio
from scikits.audiolab import Sndfile, Format

import librosa
import gamma_gvpl

# <codecell>

d = sio.loadmat('priors/lognormal_gender.mat')
U = d['U']
alpha = d['alpha'].ravel()
print U.shape, alpha.shape

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

reload(vpl)
L = alpha.size
cold_start = False

sfd = vpl.SF_Dict(np.abs(W_complex_train.T), L=L, seed=98765)
sfd.U = U.copy()
sfd.alpha = alpha.copy()

old_obj = sfd.bound()

sfd.vb_e(cold_start=cold_start, disp=0)
sfd.update_gamma(1)

score = sfd.bound()
improvement = (score - old_obj) / abs(old_obj)
print 'After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.4f}'.format(i, score, old_obj, improvement)

