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

# <codecell>

## parameters
L = 50
samples_csv = 'samples_spk_L{}.csv'.format(L)
matfile = 'spk1.mat'

# <codecell>

d = sio.loadmat(matfile)
V = d['V']
F, T = V.shape
print F, T

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

for l in xrange(L):
    figure(l)
    plot((U[l,:]))

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

reload(samples_parser)
EA, EA2, ElogA = samples_parser.parse_EA('samples_emp_L20.csv', 90, 20)

# <codecell>

specshow(EA)
colorbar()
pass

# <codecell>


