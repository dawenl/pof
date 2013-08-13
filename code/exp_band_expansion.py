# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, time

from scikits.audiolab import Sndfile, Format
import scipy.io as sio
import scipy.stats as stats

import librosa
import gap_nmf as nmf
#import gig_nmf as nmf
import sf_gap_nmf as sf_nmf
#import sf_gig_nmf as sf_nmf

import _gap

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

X_complex_train = None
for file_dir in files[:N_train]:
    for wav_dir in file_dir:
        wav, sr = load_timit(wav_dir)
        if X_complex_train is None:
            X_complex_train = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        else:
            X_complex_train = np.hstack((X_complex_train, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))) 

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

# <codecell>

subplot(211)
specshow(logspec(np.abs(X_complex_train)))
colorbar()
subplot(212)
specshow(logspec(np.abs(X_complex)))
colorbar()
pass

# <codecell>

# cut-off above 3000Hz
freq_threshold = 3000.
bin_cutoff = n_fft * freq_threshold / sr
X_cutoff = X_complex[:(bin_cutoff+1)]

x_cutoff = librosa.istft(X_cutoff, n_fft=2*bin_cutoff, hop_length=bin_cutoff, hann_w=0)
write_wav(x_cutoff, 'be_cutoff.wav', samplerate=2 * freq_threshold)

# <codecell>

d = sio.loadmat('priors/gamma_gender_batch.mat')
#d = sio.loadmat('priors/gamma_gender_full_seq.mat')
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
sfnmf = sf_nmf.SF_GaP_NMF(np.abs(X_cutoff), U[:(bin_cutoff+1)], gamma[:(bin_cutoff+1)], alpha, K=100, seed=98765)
#sfnmf = sf_nmf.SF_GIG_NMF(np.abs(X_cutoff), U[:(bin_cutoff+1)], gamma[:(bin_cutoff+1)], alpha, K=100, seed=98765)

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
#goodk = np.arange(100)
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

sfnmf.figures()

# <codecell>

K = goodk.size
fig(figsize=(16, 20))
for i, k in enumerate(goodk):
    subplot(K, 1, i+1)
    plot(np.log(sfnmf.Ew[:, k]))

# <codecell>

## Infer the high-frequency contents
Ew = np.zeros((U.shape[0], goodk.size))
for (i, k) in enumerate(goodk):
    Ew[:, i] = np.exp(np.sum(_gap.comp_log_exp(sfnmf.nua[:, k], sfnmf.rhoa[:, k], -U), axis=1))
Ew1 =  np.exp(np.dot(U, sfnmf.Ea[:, goodk]))

# <codecell>

subplot(311)
specshow(20 * np.log10(Ew[:(bin_cutoff+1)]))
colorbar()
subplot(312)
specshow(20 * np.log10(Ew1[:(bin_cutoff+1)]))
colorbar()
subplot(313)
specshow(20 * np.log10(sfnmf.Ew[:, goodk]))
colorbar()
pass

# <codecell>

subplot(211)
specshow(20 * log10(Ew))
colorbar()
subplot(212)
specshow(20 * log10(Ew1))
colorbar()
pass

# <codecell>

#c = np.mean(sfnmf.X / sfnmf._xtwid())
c = np.mean(sfnmf.X / sfnmf._xtwid(goodk))
X_bar = c * np.dot(Ew * sfnmf.Et[goodk], sfnmf.Eh[goodk])
#X_bar = c * np.dot(Ew, sfnmf.Eh[goodk])

fig()
subplot(121)
specshow(logspec(X_bar))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
subplot(122)
specshow(logspec(np.abs(X_complex)))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
pass

# <codecell>

## mean of predictive log-likelihood
pred_likeli = np.mean(stats.expon.logpdf(np.abs(X_complex[(bin_cutoff+1):]), scale=X_bar[(bin_cutoff+1):]))
print pred_likeli

# <codecell>

x_rec = librosa.istft(X_bar * (X_complex / np.abs(X_complex)), n_fft=n_fft, hann_w=0, hop_length=hop_length)
write_wav(x_rec, 'be_sf_infer.wav')

# <headingcell level=1>

# Regular NMF

# <codecell>

reload(nmf)
rnmf = nmf.GaP_NMF(np.abs(X_complex_train), K=100, seed=98765)
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

encoder = nmf.GaP_NMF(np.abs(X_cutoff), K=100, seed=98765, alpha=10.0)
encoder.rhow = rnmf.rhow[:(bin_cutoff+1)]
encoder.tauw = rnmf.tauw[:(bin_cutoff+1)]

score = -np.inf
criterion = 0.0005
for i in xrange(1000):
    encoder.update(update_w=False)
    lastscore = score
    score = encoder.bound()
    improvement = (score - lastscore) / abs(lastscore)
    print ('iteration {}: bound = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
    if improvement < criterion:
        break

# <codecell>

encoder.figures()

# <codecell>

goodk = encoder.goodk()
X_bar = np.mean(np.abs(X_cutoff)) * np.dot(rnmf.Ew[:, goodk] * encoder.Et[goodk], encoder.Eh[goodk])

# <codecell>

fig()
subplot(121)
specshow(logspec(X_bar))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
subplot(122)
specshow(logspec(np.abs(X_complex)))
axhline(y=(bin_cutoff+1), color='black')
colorbar()
pass

# <codecell>

## mean of predictive log-likelihood
pred_likeli = np.mean(stats.expon.logpdf(np.abs(X_complex[(bin_cutoff+1):]), scale=X_bar[(bin_cutoff+1):]))
print pred_likeli

# <codecell>

x_rec = librosa.istft(X_bar * (X_complex / np.abs(X_complex)), n_fft=n_fft, hann_w=0, hop_length=hop_length)
write_wav(x_rec, 'be_nmf_infer.wav')

# <codecell>


