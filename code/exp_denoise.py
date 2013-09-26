# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, time

import scipy.io as sio
import scipy.stats as stats
from scikits.audiolab import Sndfile, Format

import librosa
import kl_nmf
import beta_nmf

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

# <headingcell level=1>

# Generating Data

# <codecell>

data_mat = sio.loadmat('TIMIT_spk20.mat')
f_dirs_train = data_mat['f_dirs']
m_dirs_train = data_mat['m_dirs']

f_dirs_train = ['../../timit/train' + str(spk_dir[-10:]) for spk_dir in f_dirs_train]

# <codecell>

f_dirs_all = !ls -d ../../timit/train/dr[1-3]/f*

set_f_all = set(f_dirs_all)
set_f_train = set(f_dirs_train)

f_dirs_all = sorted(list(set_f_all.difference(set_f_train)))
#print f_dirs_all

# <codecell>

TIMIT_DIR = '../../timit/test/'

m_dirs_all = !ls -d "$TIMIT_DIR"dr[1-5]/m*

n_spk = 30
np.random.seed(98765)
f_dirs = np.random.permutation(f_dirs_all)[:n_spk]
m_dirs = np.random.permutation(m_dirs_all)[:n_spk]

f_files = {}
f_files['TIMIT_speech'] = [glob.glob(spk_dir + '/sx*.wav') for spk_dir in f_dirs]
f_files['test'] = [glob.glob(spk_dir + '/s[a|i]*.wav') for spk_dir in f_dirs]

m_files = {}
m_files['TIMIT_speech'] = [glob.glob(spk_dir + '/sx*.wav') for spk_dir in m_dirs]
m_files['test'] = [glob.glob(spk_dir + '/s[a|i]*.wav') for spk_dir in m_dirs]

# <codecell>

print f_dirs
print m_dirs

# <codecell>

print f_files
print m_files

# <codecell>

dirs = ['TIMIT_speech', 'test']

for dir_name in dirs:
    for (i, spk_dir) in enumerate(f_files[dir_name]):
        x = None
        for wav_dir in spk_dir:
            wav, sr = load_timit(wav_dir)
            if x is None:
                x = wav
            else:
                x = np.hstack((x, wav))
        write_wav(x, 'denoise/{}/sp{}.wav'.format(dir_name, i + 1))
        
    for (i, spk_dir) in enumerate(m_files[dir_name]):
        x = None
        for wav_dir in spk_dir:
            wav, sr = load_timit(wav_dir)
            if x is None:
                x = wav
            else:
                x = np.hstack((x, wav))
        write_wav(x, 'denoise/{}/sp{}.wav'.format(dir_name, i + 1 + n_spk))

# <headingcell level=1>

# Denoising

# <codecell>

n_fft = 1024
hop_length = 512

train_path = glob.glob('denoise/TIMIT_speech/*.wav')
test_path = glob.glob('denoise/test/*.wav')
noise_path = glob.glob('denoise/noise/*.wav')

print train_path
print test_path
print noise_path

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

def learn_dictionary(paths, K, d, W_nu, W_rho, W_kl, W_is, mle=True):
    for (i, path) in enumerate(paths):
        print('Learning dictionary for {}...'.format(path))
        x, sr = librosa.load(path, sr=None)
        X = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length))
        print('Bayesian NMF...')
        nmf = kl_nmf.KL_NMF(X, K=K, d=d, seed=98765)
        train(nmf)
        W_nu[i], W_rho[i] = nmf.nuw.copy(), nmf.rhow.copy()
        if mle:
            print('MLE NMF...')
            W_kl[i], _ = beta_nmf.NMF_beta(X, K, beta=1)
            W_is[i], _ = beta_nmf.NMF_beta(X, K, beta=0)
        
def compute_SNR(W, H, X_complex, s):
    Xs = np.dot(W[:, :Ks], H[:Ks])
    Xn = np.dot(W[:, Ks:], H[Ks:])
    
    Xs_rec = X_complex * Xs / (Xs + Xn)
    Xn_rec = X_complex * Xn / (Xs + Xn)
    
    xs_rec = librosa.istft(Xs_rec, n_fft=n_fft, hop_length=hop_length, hann_w=0)
    xn_rec = librosa.istft(Xn_rec, n_fft=n_fft, hop_length=hop_length, hann_w=0)

    length = min(xs_rec.size, s.size)
    snr = 10 * np.log10(np.sum( s[:length] ** 2) / np.sum( (s[:length] - xs_rec[:length])**2))
    return snr

def gen_wavs(W, H, X_complex, s_length):
    Xs = np.dot(W[:, :Ks], H[:Ks])
    Xn = np.dot(W[:, Ks:], H[Ks:])
    
    Xs_rec = X_complex * Xs / (Xs + Xn)
    Xn_rec = X_complex * Xn / (Xs + Xn)
    
    xs_rec = librosa.istft(Xs_rec, n_fft=n_fft, hop_length=hop_length, hann_w=0)
    xn_rec = librosa.istft(Xn_rec, n_fft=n_fft, hop_length=hop_length, hann_w=0)

    length = min(xs_rec.size, s_length)
    return (xs_rec[:length], xn_rec[:length])

# <codecell>

n_speech = len(train_path)
n_noise = len(noise_path)

Ks = 50
Kn = 20

d = 25

# <codecell>

#W_sf_mat = sio.loadmat('SF_dict/local/SF_TIMIT60_dict_sf_L50_TIMIT_spk20_K50_d25.mat')
W_sf_mat = sio.loadmat('SF_dict/porkpie/SF_TIMIT60_dict_sf_L50_TIMIT_spk20_K50_d25.mat')
Ws_sf_nu = W_sf_mat['W_nu']
Ws_sf_rho = W_sf_mat['W_rho']

# <codecell>

Ws_nu = np.zeros((n_speech, n_fft/2 + 1, Ks))
Ws_rho = np.zeros_like(Ws_nu)

Ws_kl = np.zeros((n_speech, n_fft/2 + 1, Ks))
Ws_is = np.zeros((n_speech, n_fft/2 + 1, Ks))

learn_dictionary(train_path, Ks, d, Ws_nu, Ws_rho, Ws_kl, Ws_is, mle=True)

# <codecell>

Wn_nu = np.zeros((n_noise, n_fft/2 + 1, Kn))
Wn_rho = np.zeros_like(Wn_nu)

Wn_kl = np.zeros((n_noise, n_fft/2 + 1, Kn))
Wn_is = np.zeros((n_noise, n_fft/2 + 1, Kn))

learn_dictionary(noise_path, Kn, d, Wn_nu, Wn_rho, Wn_kl, Wn_is, mle=True) 

# <codecell>

SNR = 0

# initialize all the results
SNR_sf = np.zeros((n_speech, n_noise))
SNR_bayes = np.zeros((n_speech, n_noise))
SNR_kl = np.zeros((n_speech, n_noise))
SNR_is = np.zeros((n_speech, n_noise))

for (i, speech) in enumerate(test_path):
    for (j, noise) in enumerate(noise_path):
        print('Speech {} under noise {}'.format(speech, noise))
        s, sr = librosa.load(speech, sr=None)
        tmp, sr = librosa.load(noise, sr=None)
        n = tmp[20 * sr: 20 * sr + s.size]
        c = sqrt( np.sum(s**2) / (np.sum(n**2) * 10**(SNR/10.)) )
        
        x = s + c*n
        X_complex = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
        X = np.abs(X_complex)
        
        xnmf_sf = kl_nmf.KL_NMF(X, K=Ks+Kn, d=d, seed=98765)
        xnmf_sf.nuw[:, :Ks], xnmf_sf.rhow[:, :Ks] = Ws_sf_nu[i], Ws_sf_rho[i]
        xnmf_sf.nuw[:, Ks:], xnmf_sf.rhow[:, Ks:] = Wn_nu[j], Wn_rho[j]
        xnmf_sf.compute_expectations()
        train(xnmf_sf, updateW=False)
        SNR_sf[i, j] = compute_SNR(xnmf_sf.Ew, xnmf_sf.Eh, X_complex, s)
        xs_sf, xn_sf = gen_wavs(xnmf_sf.Ew, xnmf_sf.Eh, X_complex, s.size)
        sio.savemat('bss/sf_s{}_n{}.mat'.format(i, j), 
                    {'se': np.vstack((xs_sf[np.newaxis, :], xn_sf[np.newaxis, :])), 
                     's': np.vstack((s[np.newaxis, :xs_sf.size], n[np.newaxis, :xn_sf.size]))})
        
        xnmf = kl_nmf.KL_NMF(X, K=Ks+Kn, d=d, seed=98765)
        xnmf.nuw[:, :Ks], xnmf.rhow[:, :Ks] = Ws_nu[i], Ws_rho[i]
        xnmf.nuw[:, Ks:], xnmf.rhow[:, Ks:] = Wn_nu[j], Wn_rho[j]
        xnmf.compute_expectations()
        train(xnmf, updateW=False)
        SNR_bayes[i, j] = compute_SNR(xnmf.Ew, xnmf.Eh, X_complex, s)
        xs, xn = gen_wavs(xnmf.Ew, xnmf.Eh, X_complex, s.size)
        sio.savemat('bss/bayes_s{}_n{}.mat'.format(i, j), 
                    {'se': np.vstack((xs[np.newaxis, :], xn[np.newaxis, :])),
                     's': np.vstack((s[np.newaxis, :xs.size], n[np.newaxis, :xn.size]))})
        
        [W_kl, H_kl] = beta_nmf.NMF_beta(X, Ks+Kn, W=np.hstack((Ws_kl[i], Wn_kl[j])), beta=1)
        SNR_kl[i, j] = compute_SNR(W_kl, H_kl, X_complex, s)
        xs_kl, xn_kl = gen_wavs(W_kl, H_kl, X_complex, s.size)
        sio.savemat('bss/kl_s{}_n{}.mat'.format(i, j), 
                    {'se': np.vstack((xs_kl[np.newaxis, :], xn_kl[np.newaxis, :])),
                     's': np.vstack((s[np.newaxis, :xs_kl.size], n[np.newaxis, :xn_kl.size]))})
        
        [W_is, H_is] = beta_nmf.NMF_beta(X, Ks+Kn, W=np.hstack((Ws_is[i], Wn_is[j])), beta=0)
        SNR_is[i, j] = compute_SNR(W_is, H_is, X_complex, s)
        xs_is, xn_is = gen_wavs(W_is, H_is, X_complex, s.size)
        sio.savemat('bss/is_s{}_n{}.mat'.format(i, j), 
                    {'se': np.vstack((xs_is[np.newaxis, :], xn_is[np.newaxis, :])),
                     's': np.vstack((s[np.newaxis, :xs_is.size], n[np.newaxis, :xn_is.size]))})

# <codecell>

#spk_IDs = ['sp01', 'sp02', 'sp03', 'sp04', 'sp05', 'sp06']
noise_names = ['Birds', 'Casino', 'Cicadas', 'Keyboard', 'Chips', 'Frogs', 'Jungle', 'Machineguns', 'Motorcycles', 'Ocean']

fig(figsize=(16, 2.5))
subplot(141)
specshow(SNR_sf)
xticks(np.arange(n_noise), noise_names, rotation=60)
#yticks(np.arange(n_speech), spk_IDs)
colorbar()
subplot(142)
specshow(SNR_bayes)
xticks(np.arange(n_noise), noise_names, rotation=60)
#yticks(np.arange(n_speech), spk_IDs)
colorbar()
subplot(143)
specshow(SNR_kl)
xticks(np.arange(n_noise), noise_names, rotation=60)
#yticks(np.arange(n_speech), spk_IDs)
colorbar()
subplot(144)
specshow(SNR_is)
xticks(np.arange(n_noise), noise_names, rotation=60)
#yticks(np.arange(n_speech), spk_IDs)
colorbar()
pass

# <codecell>

fig(figsize=(16, 10))
width = 0.3
bar(np.arange(n_noise), np.mean(SNR_sf, axis=0), width, color='r', yerr=2 * np.std(SNR_sf, axis=0)/sqrt(n_speech))
bar(np.arange(n_noise) + width, np.mean(SNR_bayes, axis=0), width, color='b', yerr=2 * np.std(SNR_bayes, axis=0)/sqrt(n_speech))
bar(np.arange(n_noise) + 2 * width, np.mean(SNR_kl, axis=0), width, color='g', yerr=2 * np.std(SNR_kl, axis=0)/sqrt(n_speech))
#bar(np.arange(n_noise) + 3 * width, np.mean(SNR_is, axis=0), width, color='y')#, yerr=2 * np.std(SNR_is, axis=0))
xticks(np.arange(n_noise), noise_names, rotation=60)
tight_layout()
#savefig('SF_vs_Bayes_vs_MLE_Ks{}_Kn{}_d{}.eps'.format(Ks, Kn, d))
pass

# <codecell>

def plot_SXR(SXR, SXR_sf, SXR_bayes, SXR_kl, SXR_is=None):
    if SXR_is is None:
        width = 0.3
    else:
        width = 0.22
    fig(figsize=(16, 10))
    bar(np.arange(n_noise), np.mean(SXR_sf, axis=0), width, color='r', yerr=2 * np.std(SXR_sf, axis=0)/sqrt(n_speech))
    bar(np.arange(n_noise) + width, np.mean(SXR_bayes, axis=0), width, color='b', yerr=2 * np.std(SXR_bayes, axis=0)/sqrt(n_speech))
    bar(np.arange(n_noise) + 2 * width, np.mean(SXR_kl, axis=0), width, color='g', yerr=2 * np.std(SXR_kl, axis=0)/sqrt(n_speech))
    if SXR_is is not None:
        bar(np.arange(n_noise) + 3 * width, np.mean(SXR_is, axis=0), width, color='y')
    xticks(np.arange(n_noise), noise_names, rotation=60)
    ylabel('{} (dB)'.format(SXR))
    tight_layout()
    #savefig('SF_vs_Bayes_vs_MLE_Ks{}_Kn{}_d{}.eps'.format(Ks, Kn, d))
    pass

SXR_mat = sio.loadmat('bss_TIMIT60_Ks50_Kn20_k25.mat')
plot_SXR('SDR', SXR_mat['SDR_sf'], SXR_mat['SDR_bayes'], SXR_mat['SDR_kl'])#, SXR_mat['SDR_is'])
plot_SXR('SIR', SXR_mat['SIR_sf'], SXR_mat['SIR_bayes'], SXR_mat['SIR_kl'])#, SXR_mat['SIR_is'])
plot_SXR('SAR', SXR_mat['SAR_sf'], SXR_mat['SAR_bayes'], SXR_mat['SAR_kl'])#, SXR_mat['SAR_is'])

# <codecell>

SDR_sf = SXR_mat['SDR_sf']
SDR_bayes = SXR_mat['SDR_bayes']
SDR_kl = SXR_mat['SDR_kl']

print np.mean(SDR_sf), np.mean(SDR_bayes), np.mean(SDR_kl)
print np.std(SDR_sf)/sqrt(SDR_sf.size), np.std(SDR_bayes)/sqrt(SDR_bayes.size), np.std(SDR_kl)/sqrt(SDR_kl.size)

print stats.ranksums(SDR_sf.ravel(), SDR_bayes.ravel())[1]
print stats.ranksums(SDR_sf.ravel(), SDR_kl.ravel())[1]
print stats.ranksums(SDR_bayes.ravel(), SDR_kl.ravel())[1]


print [stats.ranksums(SDR_sf[:, i], SDR_bayes[:, i])[1] for i in xrange(n_noise)]
print [stats.ranksums(SDR_sf[:, i], SDR_kl[:, i])[1] for i in xrange(n_noise)]

# <codecell>

diff = SDR_sf - SDR_bayes

print np.mean(diff), np.std(diff) / sqrt(diff.size)

# <codecell>


