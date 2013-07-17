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
