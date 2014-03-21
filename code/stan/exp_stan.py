# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, glob, librosa
import scipy.io as sio

from scikits.audiolab import Sndfile, Format

# <codecell>

%cd stan/
import samples_parser

# <codecell>

specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')
fig = functools.partial(figure, figsize=(16,4))

def logspec(X, amin=1e-10, dbdown=80):
    logX = 20 * np.log10(np.maximum(X, amin))
    return np.maximum(logX, logX.max() - dbdown)

# <codecell>

## parameters
L = 50
samples_csv = 'samples_spk_L{}_gamma_wp.csv'.format(L)
matfile = 'spk1.mat'

# <codecell>

d = sio.loadmat(matfile)
V = d['V']
F, T = V.shape
print F, T

# <codecell>

reload(samples_parser)
U, A, alpha, gamma = samples_parser.parse_samples(samples_csv, F, T, L)

# <codecell>

def load_timit(wav_dir):
    f = Sndfile(wav_dir, 'r')
    wav = f.read_frames(f.nframes)
    return (wav, f.samplerate)

TIMIT_DIR = '../../../timit/train/'
n_fft = 1024
hop_length = 512
speaker = True

if speaker:
    spk_dir = 'dr1/fcjf0/'
    files = glob.glob(TIMIT_DIR + spk_dir + '*.wav')

    N_train = 8
    np.random.seed(98765)
    idx = np.random.permutation(10)
    
    W_complex = None
    for file_dir in files[:N_train]:
        wav, sr = load_timit(file_dir)
        if W_complex is None:
            W_complex = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        else:
            W_complex = np.hstack((W_complex, librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))) 
else:
    wav, sr = load_timit(TIMIT_DIR + 'dr1/fcjf0/sa1.wav')
    W_complex = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)

# <codecell>

subplot(211)
specshow(U.T)
title('U')
colorbar()
subplot(212)
specshow(A.T)
title('A')
colorbar()
tight_layout()
pass

# <codecell>

meanA = np.mean(A, axis=0, keepdims=True)

tmpA = A / meanA
tmpU = U * meanA.T

subplot(211)
specshow(tmpU.T)
colorbar()
title('U')
subplot(212)
specshow(tmpA.T)
colorbar()
title('A')
tight_layout()
pass

# <codecell>

plot(np.sum(U[alpha > 1], axis=0))
pass

# <codecell>

idx = np.where(alpha > 1)[0]
for i in idx: 
    fig()
    subplot(121)
    plot(U[i])
    subplot(122)
    hist(A[:, i], bins=20)
pass

# <codecell>

for l in xrange(L):
    fig()
    subplot(121)
    plot(U[l])
    subplot(122)
    plot(np.exp(U[l]))
    tight_layout()
pass

# <codecell>

W_rec = np.exp(np.dot(U.T, A.T))
subplot(311)
specshow(logspec(W_rec))
title('UA')
colorbar()
subplot(312)
specshow(logspec(np.exp(V)))
title('log(W)')
colorbar()
subplot(313)
specshow(W_rec - np.exp(V))
title('exp(UA) - W')
colorbar()
tight_layout()
pass

# <codecell>

fig()
subplot(121)
semilogy(flipud(sort(alpha)), '-o')
subplot(122)
plot(gamma)
pass

