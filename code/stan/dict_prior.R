library(rstan)
set_cppo('fast')

# generate data
n_freq <- 20
n_time <- 20
L <- 10

seed <- 3579
set.seed(seed)
U <- matrix(rnorm(L * n_freq), nrow=L, ncol=n_freq)
alpha <- rgamma(L, shape=1)
gamma <- rgamma(n_freq, shape=100, rate=10)
A <- matrix(0, nrow=n_time, ncol=L)
V <- matrix(0, nrow=n_time, ncol=n_freq)
for (t in 1:n_time) {
  A[t,] <- rgamma(L, shape=alpha, rate=alpha)
  V[t,] <- A[t,] %*% U + rnorm(n_freq, mean=0, sd=sqrt(1/gamma))
}

posterior_approx <- '
  data {
    int<lower=0> F;
    int<lower=0> T;
    int<lower=0> L;
    matrix[T, F] V;
    matrix[L, F] U;
    vector<lower=0>[F] sigma;
    vector<lower=0>[L] alpha;
  }
  parameters {
    matrix<lower=0>[T, L] A;
  }
  model {
    for (t in 1:T) {
      A[t] ~ gamma(alpha, alpha);
      V[t] ~ normal(A[t] * U, sigma);
    }
  }
'

dat <- list(F=n_freq, T=n_time, L=L, V=V, U=U, sigma=sqrt(1/gamma), alpha=alpha)
fit <- stan(model_code = posterior_approx, data = dat, iter = 1000, chains = 4)


