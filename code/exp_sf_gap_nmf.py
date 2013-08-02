# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, pickle

from scikits.audiolab import Sndfile, Format

import librosa, gap_nmf, sf_gap_nmf
import scipy.io as sio

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

d = sio.loadmat('log_normal_gender.mat')
U = d['U'].T
gamma = d['gamma']
alpha = d['alpha'].ravel()

# <codecell>

log_normal = True

# if loading priors trained from log-normal, transfer gamma to approximate gamma noise model
if log_normal:
    gamma = 1./(np.exp(2./gamma) - np.exp(1./gamma))

# <codecell>

plot(gamma)
pass

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

# 80% of the data is used to learn the prior via empirical bayes
X_complex = None
for file_dir in files[N_train:]:
    for wav_dir in file_dir:
        wav, sr = load_timit(wav_dir)
        if X_complex is None:
            X_complex = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        #else:
        #    X_complex = np.hstack((X_complex, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))
        
X = np.abs(X_complex)

# <codecell>

specshow(logspec(X))
colorbar()
pass

# <codecell>

reload(sf_gap_nmf)
sf_gap = sf_gap_nmf.SF_GaP_NMF(X, U, gamma, alpha, K=50, seed=98765)

score = -np.inf
criterion = 0.0005
for i in xrange(5):
    sf_gap.update()
    lastscore = score
    score = sf_gap.bound()
    improvement = (score - lastscore) / np.abs(lastscore)
    print ('iteration {}: bound = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
    if improvement < criterion:
        break

# <codecell>

i

# <codecell>

fig(figsize=(16, 10))
sf_gap.figures()

# <codecell>

reload(gap_nmf)
gap = gap_nmf.GaP_NMF(X, K=50, seed=98765)

score = -np.inf
criterion = 0.0005
for i in xrange(1000):
    gap.update()
    lastscore = score
    score = gap.bound()
    improvement = (score - lastscore) / np.abs(lastscore)
    print ('iteration {}: bound = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
    if improvement < criterion:
        break

# <codecell>

fig(figsize=(16, 10))
gap.figures()

# <codecell>

X_rec_gap_amp = np.mean(X) * gap._xbar()
X_rec_gap = X_rec_gap_amp * np.exp(1j * np.angle(X_complex)) 

# <codecell>

fig(figsize=(8, 8))
subplot(311)
specshow(logspec(X))
title('Original')
colorbar()
subplot(312)
specshow(logspec(X_rec_gap_amp))
title('Reconstruction')
colorbar()
subplot(313)
specshow(X_rec_gap_amp - X)
title('Reconstruction Error')
colorbar()
tight_layout()
pass

# <codecell>

x_rec_gap = librosa.istft(X_rec_gap, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_rec_gap, 'rec_gap.wav')
x_org = librosa.istft(X_complex, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_org, 'rec_org.wav')

# <codecell>


