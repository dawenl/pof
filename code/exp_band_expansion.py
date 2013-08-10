# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, time

from scikits.audiolab import Sndfile, Format
import scipy.io as sio

import librosa
import gap_nmf as nmf
import sf_gap_nmf as sf_nmf

# <codecell>

fig = functools.partial(figure, figsize=(16,4))
specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

def logspec(X, amin=1e-10, dbdown=80):
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

TIMIT_DIR = '../../timit/train/'

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

X_complex = None
for file_dir in files[N_train:]:
    #for wav_dir in file_dir:
    #    wav, sr = load_timit(wav_dir)
    #    if X_complex is None:
    #        X_complex = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    #    else:
    #        X_complex = np.hstack((X_complex, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))
    
    ## randomly pick one sentence from each speaker
    n_wav = len(file_dir)
    wav, sr = load_timit(file_dir[np.random.choice(n_wav)])
    if X_complex is None:
        X_complex = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    else:
        X_complex = np.hstack((X_complex, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))
X = np.abs(X_complex)

# <codecell>

specshow(logspec(X))
colorbar()
print X.shape
pass

# <codecell>

# cut-off top 2 octaves (only the lowest 129 bins are kept)
X_cutoff = X_complex[:129]

x_cutoff = librosa.istft(X_cutoff, n_fft=n_fft/4, hop_length=hop_length/4, hann_w=0)
write_wav(x_cutoff, 'be_cutoff.wav', samplerate=4000)

# <codecell>

d = sio.loadmat('priors/gamma_gender_full_seq.mat')
U = d['U'].T
gamma = d['gamma']
alpha = d['alpha'].ravel()

# <codecell>

plot(gamma)
pass

# <headingcell level=1>

# Source-filter NMF

# <codecell>

reload(sf_nmf)
sfnmf = sf_nmf.SF_GaP_NMF(np.abs(X_cutoff), U[:129], gamma[:129], alpha, K=100, seed=98765)

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

goodk = sfnmf.goodk()
print goodk.shape
subplot(211)
specshow(sfnmf.Ea[:, goodk])
colorbar()
subplot(212)
specshow(logspec(sfnmf.Ew[:, goodk]))
colorbar()
pass

# <codecell>

sfnmf.figures()

# <codecell>

K = goodk.size
fig(figsize=(16, 20))
for i, k in enumerate(goodk):
    subplot(K, 1, i+1)
    plot(np.log(sfnmf.Ew[:, k]))

# <headingcell level=1>

# Regular NMF

# <codecell>

rnmf = nmf.GaP_NMF(np.abs(X_cutoff), K=100, seed=98765)

score = -np.inf
criterion = 0.0005
for i in xrange(1000):
    rnmf.update()
    lastscore = score
    score = rnmf.bound()
    improvement = (score - lastscore) / abs(lastscore)
    print ('iteration {}: bound = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
    if improvement < criterion:
        break

# <codecell>

rnmf.figures()

# <codecell>

goodk = rnmf.goodk()
K = goodk.size
fig(figsize=(16, 20))
for i, k in enumerate(goodk):
    subplot(K, 1, i+1)
    plot(np.log(rnmf.Ew[:, k]))

# <codecell>


