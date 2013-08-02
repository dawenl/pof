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

print U.shape

# <codecell>

# if loading priors trained from log-normal, transfer gamma to approximate gamma noise model
gamma = 1./(np.exp(2./gamma) - np.exp(1./gamma))

# <codecell>

plot(gamma)

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
W_complex_test = None
for file_dir in files[N_train:]:
    for wav_dir in file_dir:
        wav, sr = load_timit(wav_dir)
        if W_complex_test is None:
            W_complex_test = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        #else:
        #    W_complex_test = np.hstack((W_complex_test, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))

# <codecell>

specshow(logspec(np.abs(W_complex_test)))
colorbar()
pass

# <codecell>

reload(sf_gap_nmf)
sf_gap = sf_gap_nmf.SF_GaP_NMF(np.abs(W_complex_test), U, gamma, alpha, K=50, seed=98765)

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

gap = gap_nmf.GaP_NMF(np.abs(W_complex_test), K=50, seed=98765)

score = -np.inf
criterion = 0.0005
for i in xrange(1):
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


