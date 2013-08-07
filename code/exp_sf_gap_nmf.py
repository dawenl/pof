# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob

from scikits.audiolab import Sndfile, Format

import librosa, gap_nmf, sf_gap_nmf
import scipy.io as sio

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
        #else:
        #    X_complex = np.hstack((X_complex, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))
        
X = np.abs(X_complex)

# <codecell>

specshow(logspec(X))
colorbar()
pass

# <headingcell level=1>

# Source-filter GaP-NMF

# <codecell>

#d = sio.loadmat('priors/lognormal_gender.mat')
#d = sio.loadmat('priors/gamma_spk_stan.mat')
d = sio.loadmat('priors/gamma_gender_e15_m30_seq.mat')
U = d['U'].T
gamma = d['gamma']
alpha = d['alpha'].ravel()

# <codecell>

log_normal = False

# if loading priors trained from log-normal, transfer gamma to approximate gamma noise model
if log_normal:
    gamma = 1./(np.exp(2./gamma) - np.exp(1./gamma))

# <codecell>

plot(gamma)
pass

# <codecell>

reload(sf_gap_nmf)
sf_gap = sf_gap_nmf.SF_GaP_NMF(X, U, gamma, alpha, K=50, seed=98765)

# <codecell>

score = -np.inf
criterion = 0.0005
#objs = []
for i in xrange(10):
    #sf_gap.update(disp=1)
    sf_gap.update_h()
    sf_gap.update_w()
    goodk, _ = sf_gap.goodk()
    #print goodk
    for k in goodk:
        sf_gap.update_a(k, 0)
    sf_gap.update_theta()
    sf_gap.clear_badk()
    
    lastscore = score
    score = sf_gap.bound()
    objs.append(score)
    improvement = (score - lastscore) / np.abs(lastscore)
    print ('iteration {}: bound = {:.2f} ({:.5f} improvement)'.format(i, score, improvement))
    #if improvement < criterion:
    #    break

# <codecell>

plot(objs)
pass

# <codecell>

specshow(logspec(sf_gap.Ew))
colorbar()

# <codecell>

sf_gap.figures()

# <codecell>

_, c = sf_gap.goodk()
X_rec_sfgap_amp = c * sf_gap._xbar()
X_rec_sfgap = X_rec_sfgap_amp * np.exp(1j * np.angle(X_complex)) 

# <codecell>

x_rec_sfgap = librosa.istft(X_rec_sfgap, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_rec_sfgap, 'rec_sfgap.wav')
x_org = librosa.istft(X_complex, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_org, 'rec_org.wav')

# <headingcell level=1>

# GaP-NMF

# <codecell>

#reload(gap_nmf)
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

specshow(logspec(gap.Ew))
colorbar()

# <codecell>

gap.figures()

# <codecell>

X_rec_gap_amp = np.mean(X) * gap._xbar()
X_rec_gap = X_rec_gap_amp * np.exp(1j * np.angle(X_complex)) 

# <codecell>

x_rec_gap = librosa.istft(X_rec_gap, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_rec_gap, 'rec_gap.wav')
x_org = librosa.istft(X_complex, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(x_org, 'rec_org.wav')

# <headingcell level=1>

# SF GaP-NMF v.s. GaP-NMF

# <codecell>

fig(figsize=(16, 8))
subplot(321)
specshow(logspec(X))
title('Original')
colorbar()
subplot(323)
specshow(logspec(X_rec_sfgap_amp))
title('SF GaP-NMF')
colorbar()
subplot(324)
specshow(logspec(X_rec_gap_amp))
title('GaP-NMF')
colorbar()
subplot(325)
specshow(X_rec_sfgap_amp - X)
title('Error')
colorbar()
subplot(326)
specshow(X_rec_gap_amp - X)
title('Error')
colorbar()
tight_layout()
pass

# <codecell>


