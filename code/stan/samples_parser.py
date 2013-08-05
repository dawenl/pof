import numpy as np

# the first row is header
SKIPROWS = 1
# the number of column to be skipped (including lp__, treedepth__, and
# stepsize__)
SKIPCOLS = 3


def parse_samples(samples_csv, F, T, L, gamma_model=True, LEN_BATCH=5000):
    f = open(samples_csv, 'r')
    header = f.readline().strip().split(',')
    length = len(header) - SKIPCOLS
    f.close()

    exp_length = F * L + T * L + L + F
    print('expected # of variables: {}\tactual # of variables: {}'.format(
        exp_length, length))
    assert(exp_length == length)

    U = np.zeros((L, F))
    A = np.zeros((T, L))
    alpha = np.zeros((L,))
    if gamma_model:
        # gamma-noise model
        gamma = np.zeros((F,))
    else:
        sigma = np.zeros((F,))

    for st_idx in xrange(0, length, LEN_BATCH):
        end_idx = min(st_idx + LEN_BATCH, length)
        cols = np.arange(SKIPCOLS + st_idx, SKIPCOLS + end_idx)
        d = np.loadtxt(samples_csv, skiprows=SKIPROWS, usecols=cols,
                       delimiter=',')
        vals = np.mean(d, axis=0)
        for i, col in enumerate(cols):
            pos = header[col].strip().split('.')
            eval('assign({}, {}, {})'.format(
                pos[0], tuple([int(p) - 1 for p in pos[1:]]), vals[i]))
        print('{} variables have been processed'.format(
            cols[-1] - SKIPCOLS + 1))

    if not gamma_model:
        gamma = 1. / (sigma**2)
    return (U, A, alpha, gamma)


def parse_EA(samples_csv, T, L, LEN_BATCH=5000):
    f = open(samples_csv, 'r')
    header = f.readline().strip().split(',')
    length = len(header) - SKIPCOLS
    f.close()

    print('expected # of variables: {}\tactual # of variables: {}'.format(
        T * L, length))
    assert(T * L == length)

    EA = np.zeros((T, L))
    EA2 = np.zeros((T, L))
    ElogA = np.zeros((T, L))

    for st_idx in xrange(0, length, LEN_BATCH):
        end_idx = min(st_idx + LEN_BATCH, length)
        cols = np.arange(SKIPCOLS + st_idx, SKIPCOLS + end_idx)
        d = np.loadtxt(samples_csv, skiprows=SKIPROWS, usecols=cols,
                       delimiter=',')
        vals_EA = np.mean(d, axis=0)
        vals_EA2 = np.mean(d**2, axis=0)
        vals_ElogA = np.mean(np.log(d), axis=0)

        for i, col in enumerate(cols):
            pos = header[col].strip().split('.')
            eval('assign(EA, {}, {})'.format(
                tuple([int(p) - 1 for p in pos[1:]]), vals_EA[i]))
            eval('assign(EA2, {}, {})'.format(
                tuple([int(p) - 1 for p in pos[1:]]), vals_EA2[i]))
            eval('assign(ElogA, {}, {})'.format(
                tuple([int(p) - 1 for p in pos[1:]]), vals_ElogA[i]))
        print('{} variables have been processed'.format(
            cols[-1] - SKIPCOLS + 1))
    return (EA, EA2, ElogA)


def assign(arr, idx, val):
    arr[idx] = val
