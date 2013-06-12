library(rstan)
set_cppo('fast')

# Constant
n_freq <- 128
n_time <- 80
L <- 10

seed <- 3579
data <- gen.data(seed)
V <- data$V

maxiter <- 50
threshold <- 0.01
fit <- NULL
stanfile <- 'posterior_approx.stan'
old.obj <- -Inf

seed <- 98765
smoothness <- 100
set.seed(seed)
# Initialization
U <- matrix(rnorm(L * n_freq), nrow=L, ncol=n_freq)
alpha <- rgamma(L, shape=smoothness, rate=smoothness)
gamma <- rgamma(n_freq, shape=smoothness, rate=smoothness/2)

objs <- rep(NA, maxiter)

for (i in 1:maxiter) {
  # e-step
  fit <- e.step(stanfile, U, alpha, gamma, model=fit)
  res.exp <- comp.exp(fit)
  EA <- res.exp$EA
  EA2 <- res.exp$EA2
  ElogA <- res.exp$ElogA
  
  # m-step
  for (l in 1:L) {
    U[l,] <- update.u(l, U, EA, EA2)
  }
  gamma <- update.gamma(U, EA, EA2)
  alpha <- update.alpha(EA, ElogA)
  
  objs[i] <- comp.obj(res.exp, U, alpha, gamma)
  improvement <- (objs[i] - old.obj) / abs(objs[i])
  print(paste('After Iteration', i, 'Improvement:', improvement))
  if (improvement < threshold) {
    break
  }
  old.obj <- objs[i]
}
