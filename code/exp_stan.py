# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools
import scipy.io as sio

# <codecell>

%cd stan/
import samples_parser

# <codecell>

specshow = functools.partial(imshow, cmap=cm.hot_r, aspect='auto', origin='lower', interpolation='nearest')

# <codecell>

## parameters
samples_csv = 'samples.csv'
matfile = 'sa1.mat'
L = 40

# <codecell>

d = sio.loadmat(matfile)
V = d['V']
F, T = V.shape

# <codecell>

reload(samples_parser)
U, A, alpha, gamma = samples_parser.parse_samples(samples_csv, F, T, L)

# <codecell>

subplot(211)
specshow(U.T)
colorbar()
subplot(212)
specshow(A.T)
colorbar()

# <codecell>

V_rec = np.dot(U.T, A.T)
subplot(311)
specshow(V_rec)
colorbar()
subplot(312)
specshow(V)
colorbar()
subplot(313)
specshow(V_rec - V)
colorbar()

# <codecell>

plot(alpha)
pass

# <codecell>


