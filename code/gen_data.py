import numpy as np
import glob
import librosa
import scipy.io as sio

from scikits.audiolab import Sndfile

def load_timit(wav_dir):
    f = Sndfile(wav_dir, 'r')
    wav = f.read_frames(f.nframes)
    return (wav, f.samplerate)

TIMIT_DIR = '/home/empire6/drspeech/data/timit/dist/train/'

f_dirs_all = !ls -d "$TIMIT_DIR"dr[1-6]/f*
m_dirs_all = !ls -d "$TIMIT_DIR"dr[1-6]/m*

n_spk = 10
np.random.seed(98765)
f_dirs = np.random.permutation(f_dirs_all)[:n_spk]
m_dirs = np.random.permutation(m_dirs_all)[:n_spk]

f_files = [glob.glob(spk_dir + '/*.wav') for spk_dir in f_dirs]
m_files = [glob.glob(spk_dir + '/*.wav') for spk_dir in m_dirs]

n_fft = 1024
hop_length = 512

W_train = None
for spk_dir in f_files:
    for wav_dir in spk_dir:
        wav, _ = load_timit(wav_dir)
        if W_train is None:
            W_train = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
        else:
            W_train = np.hstack((W_train, np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))))

for spk_dir in m_files:
    for wav_dir in spk_dir:
        wav, _ = load_timit(wav_dir)
        W_train = np.hstack((W_train, np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))))

sio.savemat('TIMIT_spk{}.mat'.format(2 * n_spk), {'W': W_train, 'f_dirs': f_dirs, 'm_dirs': m_dirs})
