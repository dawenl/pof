import subprocess, sys, time

import numpy as np
import scipy.io as sio
import scipy.optimize as optimize
import scipy.special as special

import samples_parser


def gen_data(V, U, alpha, sigma, L, outfile=None):
    F, T = V.shape

    if outfile is None:
        outfile = 'emp_bayes.data.R'
    fout = open(outfile, 'w')
    fout.write('F <- {}\n'.format(F))
    fout.write('T <- {}\n'.format(T))
    fout.write('L <- {}\n'.format(L))

    write_matrix(fout, 'V', V)
    write_matrix(fout, 'U', U)
    write_array(fout, 'alpha', alpha)
    write_array(fout, 'sigma', sigma)
    fout.close()
    return outfile 

def write_matrix(fp, name, M):
    fp.write('{} <- structure(c('.format(name))
    X, Y = M.shape
    for x in xrange(X):
        for y in xrange(Y):
            fp.write(str(M[x, y]))
            if x < X-1 or y < Y-1:
                fp.write(', ')
    fp.write('), .Dim = c({}, {}))\n'.format(Y, X))
    pass

def write_array(fp, name, arr):
    fp.write('{} <- c('.format(name))
    X = len(arr)
    for i in xrange(X):
        fp.write(str(arr[i]))
        if i < X-1:
            fp.write(', ')
    fp.write(')\n')
    pass

class EBayes:
    def __init__(self, V, L=10, smoothness=100, seed=None):
        self.V = V.copy() 
        self.T, self.F = V.shape
        self.L = L
        if seed is None:
            sys.stdout.write('Using random seed\n')
            np.random.seed()
        else:
            sys.stdout.write('Using fixed seed {}\n'.format(seed))
            np.random.seed(seed) 
        sys.stdout.flush()
        self._init(smoothness=smoothness)

    def _init(self, smoothness=100):
        # model parameters
        self.U = np.random.randn(self.L, self.F)
        self.alpha = np.random.gamma(smoothness, 1./smoothness, size=(self.L,))
        self.gamma = np.random.gamma(smoothness, 2./smoothness, size=(self.F,))

    def _comp_expect(self, mu, r):
        return (np.exp(mu + 1./(2*r)), np.exp(2*mu + 2./r), mu)
         
    def e_step(self, outfile=None):
        sys.stdout.write('Variational E-step...\n')
        sys.stdout.flush()
        outfile = gen_data(self.V.T, self.U.T, self.alpha, np.sqrt(1./self.gamma), self.L, outfile=outfile)
        samples_csv = 'samples_emp_L{}.csv'.format(L) 
        subprocess.call('./posterior_approx --data={} --samples={}'.format(outfile, samples_csv).split())
        subprocess.call("grep -v '#' {} > parse_{}".format(samples_csv, samples_csv), shell=True)
        self.EA, self.EA2, self.ElogA = samples_parser.parse_EA('parse_{}'.format(samples_csv), self.T, self.L)
        pass

    def m_step(self, atol=0.01, verbose=True, disp=0):
        sys.stdout.write('Variational M-step...\n')
        sys.stdout.flush()
        old_U = self.U.copy()
        old_gamma = self.gamma.copy()
        old_alpha = self.alpha.copy()
        for l in xrange(self.L):
            self.update_u(l, disp)
        self.update_gamma()
        self.update_alpha(disp)
        self._objective_m()
        U_diff = np.mean(np.abs(self.U - old_U))
        sigma_diff = np.mean(np.abs(np.sqrt(1./self.gamma) - np.sqrt(1./old_gamma)))
        alpha_diff = np.mean(np.abs(self.alpha - old_alpha))
        if verbose:
            sys.stdout.write('U increment: {:.4f}\tsigma increment: {:.4f}\talpha increment: {:.4f}\n'.format(U_diff, sigma_diff, alpha_diff))
            sys.stdout.flush()
        if U_diff < atol and sigma_diff < atol and alpha_diff < atol:
            return True
        return False

    def update_u(self, l, disp):
        def f(u):
            return np.sum(np.outer(self.EA2[:,l], u**2) - 2*np.outer(self.EA[:,l], u) * Eres)
        
        def df(u):
            tmp = self.EA[:,l]  # for broad-casting
            return np.sum(np.outer(self.EA2[:,l], u) - Eres * tmp[np.newaxis].T, axis=0)

        Eres = self.V - np.dot(self.EA, self.U) + np.outer(self.EA[:,l], self.U[l,:])
        u0 = self.U[l,:]
        self.U[l,:], _, d = optimize.fmin_l_bfgs_b(f, u0, fprime=df, disp=0)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'U[{}, :]: {}, f={}'.format(l, d['task'], f(self.U[l,:]))
            else:
                print 'U[{}, :]: {}, f={}'.format(l, d['warnflag'], f(self.U[l,:]))

            app_grad = approx_grad(f, self.U[l,:])
            for idx in xrange(self.F):
                print 'U[{}, {:3d}] = {:.2f}\tApproximated: {:.2f}\tGradient: {:.2f}\t|Approximated - True|: {:.3f}'.format(l, idx, self.U[l,idx], app_grad[idx], df(self.U[l,:])[idx], np.abs(app_grad[idx] - df(self.U[l,:])[idx]))


    def update_gamma(self):
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.gamma = 1./np.mean(self.V**2 - 2 * self.V * EV + EV2, axis=0)

    def update_alpha(self, disp):
        def f(eta):
            tmp1 = np.exp(eta) * eta - special.gammaln(np.exp(eta))
            tmp2 = self.ElogA * (np.exp(eta) - 1) - self.EA * np.exp(eta)
            return -(self.T * tmp1.sum() + tmp2.sum())

        def df(eta):
            return -np.exp(eta) * (self.T * (eta + 1 - special.psi(np.exp(eta))) + np.sum(self.ElogA - self.EA, axis=0))
        
        eta0 = np.log(self.alpha)
        eta_hat, _, d = optimize.fmin_l_bfgs_b(f, eta0, fprime=df, disp=0)
        self.alpha = np.exp(eta_hat)
        if disp and d['warnflag']:
            if d['warnflag'] == 2:
                print 'f={}, {}'.format(f(self.alpha), d['task'])
            else:
                print 'f={}, {}'.format(f(self.alpha), d['warnflag'])
            app_grad = approx_grad(f, self.alpha)
            for l in xrange(self.L):
                print 'Alpha[{:3d}] = {:.2f}\tApproximated: {:.2f}\tGradient: {:.2f}\t|Approximated - True|: {:.3f}'.format(l, self.alpha[l], app_grad[l], df(self.alpha)[l], np.abs(app_grad[l] - df(self.alpha)[l]))


    def _objective_e(self):
        pass

    def _objective_m(self):
        self.obj = 1./2 * self.T * np.sum(np.log(self.gamma))
        EV = np.dot(self.EA, self.U)
        EV2 = np.dot(self.EA2, self.U**2) + EV**2 - np.dot(self.EA**2, self.U**2)
        self.obj -= 1./2 * np.sum((self.V**2 - 2 * self.V * EV + EV2) * self.gamma)
        self.obj += self.T * np.sum(self.alpha * np.log(self.alpha) - special.gammaln(self.alpha))
        self.obj += np.sum(self.ElogA * (self.alpha - 1) - self.EA * self.alpha)


if __name__ == '__main__':
    if len(sys.argv) != 3 and len(sys.argv) != 4: 
        print 'Usage:\n\tpython emp_bayes.py matfile L (outfile)\n\toutfile by default is emp_bayes.data.R'
        sys.exit(1)
    matfile = sys.argv[1]
    d = sio.loadmat(matfile)
    V = d['V']
    L = int(sys.argv[2])
    outfile = None
    if len(sys.argv) == 4:
        outfile = sys.argv[3]
    sfd = EBayes(V.T, L=L, seed=98765)

    # start the process
    threshold = 0.01
    old_obj = -np.inf
    maxiter = 100
    obj = []
    for i in xrange(maxiter):
        sfd.e_step(outfile)
        if sfd.m_step(disp=1):
            break
        obj.append(sfd.obj)
        improvement = (sfd.obj - old_obj) / abs(sfd.obj)
        sys.stdout.write('After ITERATION: {}\tImprovement: {:.4f}\n'.format(i, improvement))
        sys.stdout.flush()
        if (sfd.obj - old_obj) / abs(sfd.obj) < threshold:
            break
        old_obj = sfd.obj
    sio.savemat('emp_bayes.mat', {'U':sfd.U, 'alpha':sfd.alpha, 'gamma':sfd.gamma})






