# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Why do we need Taylor expansion?

# <codecell>

# original function (1 + c/k)^(-k) which happens to be the expectation of exp(a*x) if x is Gamma distributed with shape k (a is constant and c is kind of related to a)
def expect(k, c=-1.):
    return (1 + c/k)**(-k)

# first-order taylor expansion at c/k = 0 (k is large for fixed c) 
def expect_approx(c=-1.):
    return exp(-c)

# <codecell>

# let's try some huge k and see when it starts to collapse
ks = (10**np.arange(30, dtype=float64))
semilogx(ks, test(ks))
axhline(test_approx(), color='r')
pass

# <codecell>


