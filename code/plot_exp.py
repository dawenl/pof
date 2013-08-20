# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np

# <codecell>

d = np.loadtxt('pred_likelihood.out')

# <codecell>

sf_gap_pl = np.exp(d[3::3, 1])
gap_pl = np.exp(d[4::3, 1])
upper = np.exp(d[5::3, 1])
cutoff = d[3:-1:3, 0]

# <codecell>

figure(figsize=(8, 4), dpi=80)
semilogy(cutoff, upper, '--gs', linewidth=1.5, ms=5.0)
semilogy(cutoff, sf_gap_pl, '-bo', linewidth=1.5, ms=8.0)
semilogy(cutoff, gap_pl, '-r^', linewidth=1.5, ms=8.0)
xlabel('Cutoff frequency (Hz)', fontsize=12)
xlim([2900, 6100])
ylim([1e-13, 1e3])
ylabel('Geometric mean predictive likelihood', fontsize=12)
legend(['GaP-NMF\nlikelihood', 'SF-GaP-NMF', 'GaP-NMF'], loc=4, prop={'size':12})
tight_layout()
savefig('NMF_pred_likeli.eps')
pass

# <codecell>


