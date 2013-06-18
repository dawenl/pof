import sys
import numpy as np
import scipy.io as sio

SKIPROWS = 1 # the first row is header
SKIPCOLS = 3 # the number of column to be skipped (including lp__, treedepth__, and stepsize__)
LEN_BATCH = 5000

def parse_samples(samples_csv, F, T, L):
    f = open(samples_csv, 'r')
    header = f.readline().strip().split(',')
    length = len(header) - SKIPCOLS
    f.close()

    assert(F * L + T * L + L + F == length)
    
    U = np.zeros((L, F))
    A = np.zeros((T, L))
    alpha = np.zeros((L,))
    sigma = np.zeros((F,))

    for st_idx in xrange(0, length, LEN_BATCH):
        end_idx = min(st_idx + LEN_BATCH, length)
        cols = np.arange(SKIPCOLS + st_idx, SKIPCOLS + end_idx) 
        d = np.loadtxt(samples_csv, skiprows=SKIPROWS, usecols=cols, delimiter=',')
        vals = np.mean(d, axis=0)
        for i, col in enumerate(cols):
            pos = header[col].strip().split('.')
            if not header[col].startswith('U'):
                print header[col]
            eval('assign({}, {}, {})'.format(pos[0], tuple([int(p)-1 for p in pos[1:]]), vals[i]))           
        print '{} variables have been processed'.format(cols[-1] - SKIPCOLS + 1)
    gamma = 1./(sigma**2)
    return (U, A, alpha, gamma)
     

def assign(arr, idx, val):
    print idx
    print val
    arr[idx] = val


