import sys
import scipy.io as sio

def gen_data(matfile, L, outfile=None):
    d = sio.loadmat(matfile)
    V = d['V']
    F, T = V.shape

    if outfile is None:
        outfile = 'full_bayes.data.R'
    fout = open(outfile, 'w')
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
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print 'Useage\n\tpython gen_data.py matfile L (outfile)\n\toutfile = full_bayes.data.R by default'
        sys.exit(1)
    matfile = sys.argv[1]
    L = sys.argv[2]
    if len(sys.argv) == 3:
        gen_data(matfile, L)
    else:
        outfile = sys.argv[3]
        gen_data(matfile, L, outfile=outfile)

