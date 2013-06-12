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