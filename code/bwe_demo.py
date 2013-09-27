# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, pickle, time

import numpy as np
import scipy.io as sio
import scipy.stats as stats
from scikits.audiolab import Sndfile, Format

import librosa
import gamma_gvpl as vpl

import beta_nmf
import kl_nmf

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

TIMIT_DIR = '../../timit/test/'

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

f_dirs_all = !ls -d "$TIMIT_DIR"dr[1-3]/f*
m_dirs_all = !ls -d "$TIMIT_DIR"dr[1-5]/m*

n_spk = 5
np.random.seed(98765)
f_dirs = np.random.permutation(f_dirs_all)[:n_spk]
m_dirs = np.random.permutation(m_dirs_all)[:n_spk]

files = [glob.glob(spk_dir + '/*.wav')[:3] for spk_dir in f_dirs]
files.extend([glob.glob(spk_dir + '/*.wav')[:3] for spk_dir in m_dirs])

# <codecell>

print len(files)

# <codecell>

n_fft = 1024
hop_length = 512
lengths = []

train_mat = sio.loadmat('TIMIT_spk20.mat')
X_train = train_mat['W'];

X_complex_test = None
for spk_dir in files:
    for wav_dir in spk_dir:
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
specshow(logspec(X_train))
colorbar()
subplot(212)
specshow(logspec(np.abs(X_complex_test)))
colorbar()
pass

# <codecell>

# load the prior learned from training data
prior_mat = sio.loadmat('priors/sf_L50_TIMIT_spk20.mat')
U = prior_mat['U']
gamma = prior_mat['gamma'].ravel()
alpha = prior_mat['alpha'].ravel()
L = alpha.size

# <codecell>

def compute_SNR(X_complex_org, X_complex_rec, n_fft, hop_length):
    x_org = librosa.istft(X_complex_org, n_fft=n_fft, hann_w=0, hop_length=hop_length)
    x_rec = librosa.istft(X_complex_rec, n_fft=n_fft, hann_w=0, hop_length=hop_length)
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

# <codecell>

encoder_test = vpl.SF_Dict(np.abs(X_cutoff_test.T), L=L, seed=98765)
encoder_test.U, encoder_test.gamma, encoder_test.alpha = U[:, bin_low:(bin_high+1)], gamma[bin_low:(bin_high+1)], alpha

encoder_test.vb_e(cold_start = False)

# <codecell>

encoder_test = load_object('bwe_encoder_gamma')

# <codecell>

fig(figsize=(10, 6))
specshow(encoder_test.EA.T)
colorbar()
pass

# <codecell>

# plot the correlation
A_test = encoder_test.EA.copy()
A_test = A_test - np.mean(A_test, axis=0, keepdims=True)
A_test = A_test / np.sqrt(np.sum(A_test ** 2, axis=0, keepdims=True))
specshow(np.dot(A_test.T, A_test))
colorbar()
pass

# <codecell>

EX_test = np.exp(np.dot(encoder_test.EA, U)).T

# <codecell>

EexpX = np.zeros_like(np.abs(X_complex_test))
for t in xrange(encoder_test.T):
    EexpX[:, t] = np.exp(np.sum(vpl.comp_log_exp(encoder_test.a[t, :, np.newaxis], encoder_test.b[t, :, np.newaxis], U), axis=0))

# <codecell>

K = 50
W_train_kl, _ = beta_nmf.NMF_beta(X_train, K, niter=100, beta=1)

# <codecell>

def train(nmf, updateW=True, criterion=0.0005, maxiter=1000, verbose=False):
    score = nmf.bound()
    objs = []
    for i in xrange(maxiter):
        start_t = time.time()
        nmf.update(updateW=updateW, disp=1)
        t = time.time() - start_t

        lastscore = score
        score = nmf.bound()
        objs.append(score)
        improvement = (score - lastscore) / abs(lastscore)
        if verbose:
            print ('iteration {}: bound = {:.2f} ({:.5f} improvement) time = {:.2f}'.format(i, score, improvement, t))
        if i >= 10 and improvement < criterion:
            break
    return objs

# <codecell>

d = 50
nmf = kl_nmf.KL_NMF(X_train, K=K, d=d, seed=98765, U=U.T, alpha=alpha, gamma=gamma)
train(nmf, verbose=True)
pass

# <codecell>

c = nmf.X.sum() / nmf._xbar().sum()
print c
fig()
specshow(logspec(c * nmf._xbar()))
colorbar()
fig()
specshow(logspec(X_train))
colorbar()
pass

# <codecell>

specshow(logspec(nmf.Ew))
colorbar()

# <codecell>

xnmf_sf = kl_nmf.KL_NMF(np.abs(X_cutoff_test), K=K, d=d, seed=98765)
xnmf_sf.nuw, xnmf_sf.rhow = nmf.nuw[bin_low:(bin_high+1)], nmf.rhow[bin_low:(bin_high+1)]
xnmf_sf.compute_expectations()
train(xnmf_sf, updateW=False)
pass

# <codecell>

c = xnmf_sf.X.sum() / xnmf_sf._xbar().sum()
print c

fig()
specshow(logspec(c * xnmf_sf._xbar()))
colorbar()

fig()
specshow(logspec(xnmf_sf.X))
colorbar()

pass

# <codecell>

c = xnmf_sf.X.sum() / xnmf_sf._xbar().sum()
print c
EX_SF_NMF = c * np.dot(nmf.Ew, xnmf_sf.Eh)

# <codecell>

fig()
specshow(logspec(EX_SF_NMF))
colorbar()
pass

# <codecell>

_, H_test_kl = beta_nmf.NMF_beta(np.abs(X_cutoff_test), K, niter=10, W=W_train_kl[bin_low:(bin_high+1), :], beta=1)
EX_KL = np.dot(W_train_kl, H_test_kl)

# <codecell>

threshold = np.amax(tmpX)
EX_test[EX_test >= threshold] = threshold
EX_KL[EX_KL >= threshold] = threshold

# <codecell>

freq_res = sr / n_fft

fig(figsize=(12, 3))
specshow(logspec(np.abs(X_complex_test)))
axhline(y=(bin_low+1), color='black')
axhline(y=(bin_high+1), color='black')
ylabel('Frequency (Hz)')
#yticks(arange(0, 513, 100), freq_res * arange(0, 513, 100))
xlabel('Time (sec)')
#xticks(arange(0, 2600, 500), (float(hop_length) / sr * arange(0, 2600, 500)))
colorbar()
tight_layout()
#savefig('bwe_org.eps')

#fig(figsize=(12, 3))
#specshow(logspec(tmpX))
#ylabel('Frequency (Hz)')
#yticks(arange(0, 513, 100), freq_res * arange(0, 513, 100))
#xlabel('Time (sec)')
#xticks(arange(0, 2600, 500), (float(hop_length) / sr * arange(0, 2600, 500)))
#colorbar()
#tight_layout()
#savefig('bwe_cutoff.eps')

fig(figsize=(12, 3))
specshow(logspec(EX_test))
axhline(y=(bin_low+1), color='black')
axhline(y=(bin_high+1), color='black')
ylabel('Frequency (Hz)')
#yticks(arange(0, 513, 100), freq_res * arange(0, 513, 100))
xlabel('Time (sec)')
#xticks(arange(0, 2600, 500), (float(hop_length) / sr * arange(0, 2600, 500)))
colorbar()
tight_layout()
#savefig('bwe_rec.eps')

fig(figsize=(12, 3))
specshow(logspec(EX_KL))
axhline(y=(bin_low+1), color='black')
axhline(y=(bin_high+1), color='black')
ylabel('Frequency (Hz)')
#yticks(arange(0, 513, 100), freq_res * arange(0, 513, 100))
xlabel('Time (sec)')
#xticks(arange(0, 2600, 500), (float(hop_length) / sr * arange(0, 2600, 500)))
colorbar()
tight_layout()
#savefig('bwe_kl_rec.eps')

fig(figsize=(12, 3))
specshow(logspec(EX_SF_NMF))
axhline(y=(bin_low+1), color='black')
axhline(y=(bin_high+1), color='black')
ylabel('Frequency (Hz)')
#yticks(arange(0, 513, 100), freq_res * arange(0, 513, 100))
xlabel('Time (sec)')
#xticks(arange(0, 2600, 500), (float(hop_length) / sr * arange(0, 2600, 500)))
colorbar()
tight_layout()
pass

# <codecell>

pos = np.cumsum(lengths)

SNR_KL = np.zeros((pos.size, ))
start_pos = 0
for (i, p) in enumerate(pos):
    _, x_rec, SNR_KL[i] = compute_SNR(X_complex_test[:, start_pos:p], 
                                  EX_KL[:, start_pos:p] * (X_complex_test[:, start_pos:p] / np.abs(X_complex_test[:, start_pos:p])), 
                                  n_fft, hop_length)
    write_wav(x_rec, 'bwe/{}_kl_rec.wav'.format(i+1))
    start_pos = p
print 'SNR = {:.3f} +- {:.3f}'.format(np.mean(SNR_KL), 2*np.std(SNR_KL)/sqrt(pos.size))
print SNR_KL

# <codecell>

SNR_SF_NMF = np.zeros((pos.size, ))
start_pos = 0
for (i, p) in enumerate(pos):
    _, x_rec, SNR_SF_NMF[i] = compute_SNR(X_complex_test[:, start_pos:p], 
                                  EX_SF_NMF[:, start_pos:p] * (X_complex_test[:, start_pos:p] / np.abs(X_complex_test[:, start_pos:p])), 
                                  n_fft, hop_length)
    write_wav(x_rec, 'bwe/{}_sfnmf_rec.wav'.format(i+1))
    start_pos = p
print 'SNR = {:.3f} +- {:.3f}'.format(np.mean(SNR_SF_NMF), 2*np.std(SNR_SF_NMF)/sqrt(pos.size))
print SNR_SF_NMF

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

x_test_org, x_test_rec, SNR_SF_all = compute_SNR(X_complex_test, EX_test * (X_complex_test / np.abs(X_complex_test)), 
                                                 n_fft, hop_length)
_, x_test_rec_kl, SNR_KL_all = compute_SNR(X_complex_test, EX_KL * (X_complex_test / np.abs(X_complex_test)), 
                                                 n_fft, hop_length)
_, x_test_rec_sfnmf, SNR_SFNMF_all = compute_SNR(X_complex_test, EX_SF_NMF * (X_complex_test / np.abs(X_complex_test)), 
                                                 n_fft, hop_length)
print SNR_SF_all
print SNR_KL_all
print SNR_SFNMF_all

# <codecell>

write_wav(x_test_rec, 'bwe_demo_rec.wav')
write_wav(x_test_org, 'bwe_demo_org.wav')
write_wav(x_test_rec_kl, 'bwe_demo_rec_kl.wav')

# <codecell>

save_object(encoder_test, 'bwe_encoder_gamma')

# <codecell>

ovl_mat = sio.loadmat('bwe_ovl.mat')
sig_sf = ovl_mat['Csig_sf']
ovl_sf = ovl_mat['Covl_sf']

sig_kl = ovl_mat['Csig_kl']
ovl_kl = ovl_mat['Covl_kl']

sig_cutoff = ovl_mat['Csig_cutoff']
ovl_cutoff = ovl_mat['Covl_cutoff']

# <codecell>

print np.mean(sig_sf), 2 * np.std(sig_sf) / sqrt(pos.size)
print np.mean(sig_kl), 2 * np.std(sig_kl) / sqrt(pos.size)
print np.mean(sig_cutoff), 2 * np.std(sig_cutoff) / sqrt(pos.size)

# <codecell>

print np.mean(ovl_sf), 2 * np.std(ovl_sf) / sqrt(pos.size)
print np.mean(ovl_kl), 2 * np.std(ovl_kl) / sqrt(pos.size)
print np.mean(ovl_cutoff), 2 * np.std(ovl_cutoff) / sqrt(pos.size)

# <codecell>


