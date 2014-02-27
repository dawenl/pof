# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import glob
import itertools

import librosa
import scipy.io as sio
from scikits.audiolab import Sndfile

# <codecell>

def load_timit(wav_dir):
    f = Sndfile(wav_dir, 'r')
    wav = f.read_frames(f.nframes)
    return (wav, f.samplerate)

TIMIT_DIR = '../../timit/'

n_mspk = 70
n_fspk = 70

# <codecell>

n_dr = 8
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
    ftmp = !ls -d "$TIMIT_DIR"train/"$dr"/f*
    mtmp = !ls -d "$TIMIT_DIR"train/"$dr"/m*
    f_dirs.extend(np.random.choice(ftmp, fspk_dict[dr]))
    m_dirs.extend(np.random.choice(mtmp, mspk_dict[dr]))

# <codecell>

files, phones = [], []

for spk_dir in itertools.chain(f_dirs, m_dirs):
    files.extend(glob.glob(spk_dir + '/s[a|i|x]*.wav'))
    phones.extend(glob.glob(spk_dir + '/s[a|i|x]*.phn'))

# <codecell>

print f_dirs
print m_dirs

# <codecell>

len(files)

# <codecell>

map_64to48 = {}
with open(TIMIT_DIR + '64to48.map', 'r') as outfile:
    for line in outfile:
        k, v = line.strip().split()
        map_64to48[k] = v
        
map_48to39 = {}
with open(TIMIT_DIR + '48to39.map', 'r') as outfile:
    for line in outfile:
        k, v = line.strip().split()
        map_48to39[k] = v

# <codecell>

map_64to39 = {}
for (k, v) in map_64to48.items():
    print 'map "%s" to "%s" then to "%s"' % (k, v, map_48to39[v]) 
    map_64to39[k] = map_48to39[v]

# <codecell>

fs = 16000
n_fft = 1024
hop_length = 512

W_train = None
for wav_dir in files:
    wav, _ = load_timit(wav_dir)
    if W_train is None:
        W_train = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
    else:
        W_train = np.hstack((W_train, np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))))

# <codecell>

W_train.shape

# <codecell>

sio.savemat('TIMIT_fspk%d_mspk%d_F%d_H%d.mat' % (n_fspk, n_mspk, n_fft, hop_length), {'W': W_train, 'files': files})

# <codecell>


