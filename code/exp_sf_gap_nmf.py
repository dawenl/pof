# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import librosa, sf_gap_nmf
import scipy.io as sio

# <codecell>

d = sio.loadmat('log_normal_gender.mat')
U = d['U']
gamma = d['gamma']
alpha = d['alpha']

# <codecell>

# if loading priors trained from log-normal, transfer gamma to approximate gamma noise model
gamma = 

