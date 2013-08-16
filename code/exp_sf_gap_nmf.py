# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, time

import scipy.io as sio
from scikits.audiolab import Sndfile, Format

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

# 80% of the data is used to learn the prior via empirical bayes
X_complex = None
for file_dir in files[N_train:]:
    for wav_dir in file_dir:
        wav, sr = load_timit(wav_dir)
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

# <headingcell level=1>

# Source-filter NMF

# <codecell>

#d = sio.loadmat('priors/lognormal_gender.mat')
#d = sio.loadmat('priors/gamma_spk_stan.mat')
#d = sio.loadmat('priors/gamma_gender_e15_m30_seq.mat')
d = sio.loadmat('priors/gamma_gender_full_seq.mat')
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
print amax(U), amin(U)
pass

# <codecell>

reload(sf_nmf)
sfnmf = sf_nmf.SF_GaP_NMF(X, U, gamma, alpha, K=100, seed=98765)

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

subplot(211)
specshow(sfnmf.Ea)
colorbar()
subplot(212)
specshow(logspec(sfnmf.Ew))
colorbar()
pass

# <codecell>

sfnmf.figures()

# <codecell>

goodk = sfnmf.goodk()
c = np.mean(sfnmf.X / sfnmf._xtwid(goodk))
#c = np.mean(sfnmf.X / sfnmf._xtwid())
X_rec_sf_amp = c * sfnmf._xbar()
X_rec_sf = X_rec_sf_amp * X_complex / np.abs(X_complex)

# <codecell>

K = goodk.size
for (i, k) in enumerate(goodk):
    fig(figsize=(12, 2))
    plot(((sfnmf.Ew[:, k])))
    xlim([0, 513])

# <codecell>

x_rec_sf = librosa.istft(X_rec_sf, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_rec_sf, 'rec_sf.wav')
x_org = librosa.istft(X_complex, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_org, 'rec_org.wav')

# <headingcell level=1>

# Regular NMF

# <codecell>

reload(nmf)
rnmf = nmf.GaP_NMF(X, K=50, seed=98765)

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

specshow(logspec(rnmf.Ew))
colorbar()
pass

# <codecell>

rnmf.figures()

# <codecell>

#c = np.mean(rnmf.X / rnmf._xtwid())
#X_rec_amp = c * rnmf._xbar()
X_rec_amp = np.mean(X) * rnmf._xbar()
X_rec = X_rec_amp * X_complex / np.abs(X_complex)

# <codecell>

x_rec = librosa.istft(X_rec, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_rec, 'rec_nmf.wav')
x_org = librosa.istft(X_complex, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_org, 'rec_org.wav')

# <headingcell level=1>

# SF-NMF v.s. Regular NMF

# <codecell>

fig(figsize=(16, 8))
subplot(321)
specshow(logspec(X))
title('Original')
colorbar()
subplot(323)
specshow(logspec(X_rec_sf_amp))
title('SF NMF')
colorbar()
subplot(324)
specshow(logspec(X_rec_amp))
title('NMF')
colorbar()
subplot(325)
specshow(X_rec_sf_amp - X)
title('Error')
colorbar()
subplot(326)
specshow(X_rec_amp - X)
title('Error')
colorbar()
tight_layout()
pass

# <codecell>


