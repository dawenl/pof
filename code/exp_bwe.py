# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, itertools, time
import cPickle as pickle

import numpy as np
import scipy.io as sio
import scipy.stats as stats
from scikits.audiolab import Sndfile, Format

import librosa
from librosa.display import *
import npof as pof

# <codecell>

%cd nmf/
import beta_nmf
import kl_nmf
%cd ..

# <codecell>

fig = functools.partial(figure, figsize=(16,4))

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

TIMIT_DIR = '../../timit/test'

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

n_dr = 8

n_mspk = 10
n_fspk = 10
drs = ['dr' + str(i) for i in xrange(1, n_dr+1)]

fspk_dict = dict.fromkeys(drs, n_fspk/n_dr)
mspk_dict=  dict.fromkeys(drs, n_mspk/n_dr)

np.random.seed(12345)
for (mk, fk) in zip(np.random.choice(drs, size=n_mspk % n_dr, replace=False), 
                    np.random.choice(drs, size=n_fspk % n_dr, replace=False)):
    mspk_dict[mk] += 1
    fspk_dict[fk] += 1

# <codecell>

print fspk_dict
print mspk_dict

# <codecell>

f_dirs, m_dirs = [], []
for dr in drs:
    ftmp = !ls -d "$TIMIT_DIR"/"$dr"/f*
    mtmp = !ls -d "$TIMIT_DIR"/"$dr"/m*
    f_dirs.extend(np.random.choice(ftmp, fspk_dict[dr]))
    m_dirs.extend(np.random.choice(mtmp, mspk_dict[dr]))
    
files = []

for spk_dir in itertools.chain(f_dirs, m_dirs):
    files.extend(sorted(glob.glob(spk_dir + '/*.wav'))[:3])

# <codecell>

print len(files)

# <codecell>

n_fft = 1024
hop_length = 512
lengths = []

train_mat = sio.loadmat('TIMIT_fspk50_mspk50_F1024_H512.mat')
X_train = train_mat['W'];

X_complex_test = None
for wav_dir in files:
    wav, sr = load_timit(wav_dir)
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
    lengths.append(stft.shape[1])
    if X_complex_test is None:
        X_complex_test = stft
    else:
        X_complex_test = np.hstack((X_complex_test, stft))

# <codecell>

fig()
subplot(211)
specshow(logspec(X_train), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
colorbar()
subplot(212)
specshow(logspec(np.abs(X_complex_test)), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
colorbar()
tight_layout()
pass

# <codecell>

pof_params = sio.loadmat('priors/sf_L50_TIMIT_spk100_F1024_H512.mat')
U = pof_params['U']
gamma = pof_params['gamma'].ravel()
alpha = pof_params['alpha'].ravel()
L = alpha.size

# <codecell>

def compute_SNR(X_complex_org, X_complex_rec, n_fft, hop_length):
    x_org = librosa.istft(X_complex_org, hop_length=hop_length, window=np.ones((n_fft, )))
    x_rec = librosa.istft(X_complex_rec, hop_length=hop_length, window=np.ones((n_fft, )))
    length = min(x_rec.size, x_org.size)
    snr = 10 * np.log10(np.sum( x_org[:length] ** 2) / np.sum( (x_org[:length] - x_rec[:length])**2))
    return (x_org, x_rec, snr)

# <codecell>

# only keep the contents between 400-3400 Hz
freq_high = 3400
freq_low = 400
bin_high = n_fft * freq_high / sr
bin_low = n_fft * freq_low / sr
X_cutoff_test = X_complex_test[bin_low:(bin_high+1)]

# <codecell>

F, T = X_complex_test.shape
tmpX = np.zeros((F, T))
tmpX[bin_low:(bin_high+1)] = np.abs(X_cutoff_test)

# <headingcell level=1>

# BWE with PoF

# <codecell>

encoder = pof.ProductOfFiltersLearning(n_feats=F, n_filters=L, 
                                       U=U[:, bin_low:(bin_high+1)], gamma=gamma[bin_low:(bin_high+1)], alpha=alpha, 
                                       n_jobs=5, random_state=98765, verbose=True)
EA = encoder.transform(np.abs(X_cutoff_test.T))

# <codecell>

fig(figsize=(10, 6))
specshow(EA.T)
colorbar()
pass

# <codecell>

# plot the correlation
A_test = EA.copy()
A_test = A_test - np.mean(A_test, axis=0, keepdims=True)
A_test = A_test / np.sqrt(np.sum(A_test ** 2, axis=0, keepdims=True))
specshow(np.dot(A_test.T, A_test))
colorbar()
pass

# <codecell>

EX_test = np.exp(np.dot(EA, U)).T
EX_test[bin_low:(bin_high+1)] = np.abs(X_cutoff_test)
EX_test[EX_test > tmpX.max()] = tmpX.max()

# <headingcell level=1>

# BWE with NMF

# <codecell>

K = 10

# <codecell>

reload(beta_nmf)
W_train_kl, _ = beta_nmf.NMF_beta(X_train, K, maxiter=500, beta=1, criterion=0)

# <codecell>

d = 50
SF_NMF = False

if SF_NMF:
    dict_mat = sio.loadmat('SF_dict/porkpie/SF_BWE_K{}_d{}.mat'.format(K, d))
    nuw = dict_mat['W_nu']
    rhow = dict_mat['W_rho']
    Ew = nuw / rhow
else:
    nmf = kl_nmf.KL_NMF(X_train, K=K, d=d, seed=98765)
    train(nmf, criterion=0.0001, verbose=True)
    nuw, rhow = nmf.nuw, nmf.rhow
    Ew = nmf.Ew

# <codecell>

fig()
subplot(121)
specshow(logspec(Ew))
colorbar()
subplot(122)
specshow(logspec(W_train_kl))
colorbar()
pass

# <codecell>

xnmf_sf = kl_nmf.KL_NMF(np.abs(X_cutoff_test), K=K, d=d, seed=98765)
xnmf_sf.nuw, xnmf_sf.rhow = nuw[bin_low:(bin_high+1)], rhow[bin_low:(bin_high+1)]
xnmf_sf.compute_expectations()
train(xnmf_sf, updateW=False, criterion=0.0001, verbose=True)
pass

# <codecell>

c = xnmf_sf.X.sum() / xnmf_sf._xbar().sum()
EX_SF_NMF = c * np.dot(Ew, xnmf_sf.Eh)
EX_SF_NMF[bin_low:(bin_high+1)] = np.abs(X_cutoff_test)

# <codecell>

fig()
specshow(logspec(EX_SF_NMF))
colorbar()
pass

# <codecell>

#reload(beta_nmf)
_, H_test_kl = beta_nmf.NMF_beta(np.abs(X_cutoff_test), K, maxiter=50, 
                                 W=W_train_kl[bin_low:(bin_high+1), :], beta=1, criterion=0)
EX_KL = np.dot(W_train_kl, H_test_kl)
EX_KL[bin_low:(bin_high+1)] = np.abs(X_cutoff_test)

# <codecell>

freq_res = sr / n_fft

fig(figsize=(12, 3))
specshow(logspec(np.abs(X_complex_test)), cmap=cm.hot_r, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
axhline(y=(bin_low+1), color='black')
axhline(y=(bin_high+1), color='black')
colorbar()
tight_layout()

fig(figsize=(12, 3))
specshow(logspec(EX_test), cmap=cm.hot_r, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
axhline(y=(bin_low+1), color='black')
axhline(y=(bin_high+1), color='black')
colorbar()
tight_layout()

fig(figsize=(12, 3))
specshow(logspec(EX_KL), cmap=cm.hot_r, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear')
axhline(y=(bin_low+1), color='black')
axhline(y=(bin_high+1), color='black')
colorbar()
tight_layout()

# <codecell>

pos = np.cumsum(lengths)

SNR_KL = np.zeros((pos.size, ))
start_pos = 0
for (i, p) in enumerate(pos):
    x_org, x_rec, SNR_KL[i] = compute_SNR(X_complex_test[:, start_pos:p], 
                                  EX_KL[:, start_pos:p] * (X_complex_test[:, start_pos:p] / np.abs(X_complex_test[:, start_pos:p])), 
                                  n_fft, hop_length)
    write_wav(x_org, 'bwe/{}_org.wav'.format(i+1))
    write_wav(x_rec, 'bwe/{}_kl_rec.wav'.format(i+1))
    start_pos = p
print 'SNR = {:.3f} +- {:.3f}'.format(np.mean(SNR_KL), 2*np.std(SNR_KL)/sqrt(pos.size))
print SNR_KL

# <codecell>

SNR_SF = np.zeros((pos.size, ))
start_pos = 0
for (i, p) in enumerate(pos):
    x_org, x_rec, SNR_SF[i] = compute_SNR(X_complex_test[:, start_pos:p], 
                                  EX_test[:, start_pos:p] * (X_complex_test[:, start_pos:p] / np.abs(X_complex_test[:, start_pos:p])), 
                                  n_fft, hop_length)
    write_wav(x_org, 'bwe/{}_org.wav'.format(i+1))
    write_wav(x_rec, 'bwe/{}_sf_rec.wav'.format(i+1))
    start_pos = p
print 'SNR = {:.3f} +- {:.3f}'.format(np.mean(SNR_SF), 2*np.std(SNR_SF)/sqrt(pos.size))
print SNR_SF

# <codecell>

tmpX_complex = np.zeros((F, T), dtype=complex)
tmpX_complex[bin_low:(bin_high+1)] = X_cutoff_test

tmp1, tmp2, SNR_init = compute_SNR(X_complex_test, tmpX_complex, n_fft, hop_length)
print SNR_init

SNR_cutoff = np.zeros((pos.size, ))
start_pos = 0
for (i, p) in enumerate(pos):
    x_org, x_rec, SNR_cutoff[i] = compute_SNR(X_complex_test[:, start_pos:p], tmpX_complex[:, start_pos:p], n_fft, hop_length)
    write_wav(x_rec, 'bwe/{}_cutoff.wav'.format(i+1))
    start_pos = p
print 'SNR = {:.3f} +- {:.3f}'.format(np.mean(SNR_cutoff), 2*np.std(SNR_cutoff)/sqrt(pos.size))
print SNR_cutoff

# <codecell>

EX_rand = tmpX.copy()
EX_rand[:bin_low] = tmpX.max() * np.random.rand(bin_low, tmpX.shape[1])
EX_rand[bin_high+1:] = tmpX.max() * np.random.rand(tmpX.shape[0] - bin_high-1 ,tmpX.shape[1])

# <codecell>

SNR_rand = np.zeros((pos.size, ))
start_pos = 0
for (i, p) in enumerate(pos):
    x_org, x_rec, SNR_rand[i] = compute_SNR(X_complex_test[:, start_pos:p], 
                                  EX_rand[:, start_pos:p] * (X_complex_test[:, start_pos:p] / np.abs(X_complex_test[:, start_pos:p])), 
                                  n_fft, hop_length)
    #write_wav(x_org, 'bwe/{}_org.wav'.format(i+1))
    write_wav(x_rec, 'bwe/{}_rand.wav'.format(i+1))
    start_pos = p
print 'SNR = {:.3f} +- {:.3f}'.format(np.mean(SNR_rand), 2*np.std(SNR_rand)/sqrt(pos.size))
print SNR_rand

