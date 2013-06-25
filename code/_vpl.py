import numpy as np
import scipy.special as special

# E[x], E[x^2], E[log(x)] for x under different distributions
def exp_lognormal(mu, sigma):
    return (np.exp(mu + sigma/2), np.exp(2*mu + 2*sigma), mu)

def entropy_lognorml(mu, sigma):
    return (np.log(sigma)/2 + mu)


def exp_gamma(alpha, mu):
    return (mu, mu**2 + mu**2/alpha, special.psi(alpha) - np.log(alpha/mu))

def entropy_gamma(alpha, mu):
    return (alpha - np.log(alpha / mu) + special.gammaln(alpha) + (1-alpha) * special.psi(alpha))
