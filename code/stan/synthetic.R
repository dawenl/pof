# Generate Synthetic data
gen.data <- function(seed) {
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
  return (list(V=V, A=A, U=U, alpha=alpha, gamma=gamma))
}

# Constant
n_freq <- 20
n_time <- 25
L <- 10

seed <- 3579
data <- gen.data(seed)
V <- data$V
