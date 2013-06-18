# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools, librosa
import scipy.io as sio

from scikits.audiolab import Sndfile, Format

# <codecell>

%cd stan/
import samples_parser

# <codecell>

specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

# <codecell>

## parameters
L = 40
samples_csv = 'samples_L{}.csv'.format(L)
matfile = 'sa1.mat'

# <codecell>

d = sio.loadmat(matfile)
V = d['V']
F, T = V.shape

# <codecell>

U, A, alpha, gamma = samples_parser.parse_samples(samples_csv, F, T, L)

# <codecell>

def load_timit(wav_dir):
    f = Sndfile(wav_dir, 'r')
    wav = f.read_frames(f.nframes)
    return (wav, f.samplerate)

TIMIT_DIR = '../../../timit/train/'
n_fft = 1024
hop_length = 512
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

V_rec = np.dot(U.T, A.T)
subplot(311)
specshow(V_rec)
title('UA')
colorbar()
subplot(312)
specshow(V)
title('log(W)')
colorbar()
subplot(313)
specshow(np.exp(V_rec) - np.exp(V))
title('exp(UA) - W')
colorbar()
tight_layout()
pass

# <codecell>

def write_wav(w, filename, channels=1, samplerate=16000):
    f_out = Sndfile(filename, 'w', format=Format(), channels=channels, samplerate=samplerate)
    f_out.write_frames(w)
    f_out.close()
    pass

W_rec = np.exp(V_rec) * np.exp(1j * np.angle(W_complex))
wav_rec = librosa.istft(W_rec, n_fft=n_fft, hop_length=hop_length, hann_w=0)
write_wav(wav_rec, 'rec_stan_L{}_F{}_H{}.wav'.format(L, n_fft, hop_length))

# <codecell>

plot(flipud(sort(alpha)), '-o')
pass

# <codecell>


