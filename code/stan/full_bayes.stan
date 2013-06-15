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

