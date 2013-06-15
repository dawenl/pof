import sys
import scipy.io as sio

def gen_data(matfile, L):
    d = sio.loadmat(matfile)
    V = d['V']
    F, T = V.shape

    fout = open('full_bayes.data.R', 'w')
    fout.write('F <- {}\n'.format(F))
    fout.write('T <- {}\n'.format(T))
    fout.write('L <- {}\n'.format(L))
    fout.write('V <- structure(c(')
    for f in xrange(F):
        for t in xrange(T):
            fout.write(str(V[f,t]))
            if f < F-1 or t < T-1:
                fout.write(', ')
    fout.write('), .Dim = c({}, {}))\n'.format(T, F))
    fout.close()
    pass

if __name__ == '__main__':
    matfile = sys.argv[1]
    L = sys.argv[2]
    gen_data(matfile, L)
