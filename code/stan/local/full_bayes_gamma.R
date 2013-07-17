library(rstan)
set_cppo('fast')

full_bayes <- '
data {
    int<lower=0> F;
    int<lower=0> T;
    int<lower=0> L;
    matrix<lower=0>[T, F] V;
  }
  parameters {
    matrix[L, F] U;
    row_vector<lower=0>[F] gamma;
    vector<lower=0>[L] alpha;
    matrix<lower=0>[T, L] A;
  }
  model {
    gamma ~ exponential(10);
    alpha ~ exponential(10);
    for (l in 1:L) {
      U[l] ~ normal(0, 4);
    }
    for (t in 1:T) {
      A[t] ~ gamma(alpha, alpha);
      V[t] ~ gamma(gamma, gamma ./ exp(A[t] * U));
    }
  }
'

dat <- list(F=n_freq, T=n_time, L=L, V=V)
fit <- stan(model_code=full_bayes, data=dat, iter=1000, chains=1)
pars <- extract(fit, permuted=T)

U.sample <- matrix(pars$U, nrow=n_freq * L, ncol=500, byrow=T)
U.fb <- matrix(apply(U.sample, 1, mean), nrow=L, ncol=n_freq)

A.sample <- matrix(pars$A, nrow=n_time * L, ncol=500, byrow=T)
A.fb <- matrix(apply(A.sample, 1, mean), nrow=n_time, ncol=L)

alpha.sample <- matrix(pars$alpha, nrow=L, ncol=500, byrow=T)
alpha.fb <- apply(alpha.sample, 1, mean)

gamma.sample <- matrix(pars$gamma, nrow=n_freq, ncol=500, byrow=T)
gamma.fb <- apply(gamma.sample, 1, mean)
