library(rstan)
set_cppo('fast')

full_bayes <- '
  data {
      int<lower=0> F;
      int<lower=0> T;
      int<lower=0> L;
      matrix[T, F] V;
    }
    parameters {
      matrix[L, F] U;
      vector<lower=0>[F] sigma;
      vector<lower=0>[L] alpha;
      matrix<lower=0>[T, L] A;
    }
    model {
      sigma ~ exponential(10);
      alpha ~ exponential(10);
      for (l in 1:L) {
        U[l] ~ normal(0, 100);
      }
      for (t in 1:T) {
        A[t] ~ gamma(alpha, alpha);
        V[t] ~ normal(A[t] * U, sigma);
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

sigma.sample <- matrix(pars$sigma, nrow=n_freq, ncol=500, byrow=T)
sigma.fb <- apply(sigma.sample, 1, mean)

full.bayes <- list(U=U.fb, A=A.fb, alpha=alpha.fb, sigma=sigma.fb)
save(full.bayes, file='full_bayes.RData')

