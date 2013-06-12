library(rstan)
set_cppo('fast')

# Constant
n_freq <- 20
n_time <- 20
L <- 10
seed <- 3579

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

# E-step
e.step <- function(stanfile, U, alpha, gamma, model=NULL) {
  dat <- list(F=n_freq, T=n_time, L=L, V=V, U=U, sigma=sqrt(1/gamma), alpha=alpha)
  if (is.null(model)) {
    fit <- stan(file=stanfile, data=dat, iter=1000, chains=4)
  } else {
    fit <- stan(fit=model, data=dat, iter=1000, chains=4)
  } 
  return (fit)
}

comp.exp <- function(fit) {
  A.sample <- matrix(extract(fit, permuted=T)$A, nrow=n_time * L, ncol=2000, byrow=T)
  EA <- matrix(apply(A.sample, 1, mean), nrow=n_time, ncol=L, byrow=T)
  EA2 <- matrix(apply(A.sample^2, 1, mean), nrow=n_time, ncol=L, byrow=T)
  ElogA <- matrix(apply(log(A.sample), 1, mean), nrow=n_time, ncol=L, byrow=T)
  return (list(EA=EA, EA2=EA2, ElogA=ElogA))
}

# M-step
m.step <- function(res.exp) {
  EA <- res.exp$EA
  EA2 <- res.exp$EA2
  ElogA <- res.exp$ElogA
  
  U.res <- matrix(0, nrow=L, ncol=n_freq)
  for (l in 1:L) {
    U.res[l,] <- update.u(l, EA, EA2)
  }
  
  gamma.res <- update.gamma(EA, EA2)
  alpha.res <- update.alpha(EA, ElogA)
  
  return (list(U=U.res, gamma=gamma.res, alpha=alpha.res))
}

update.u <- function(l, EA, EA2) { 
  f.u <- function(u) {
    res <- sum(outer(EA2[,l], u**2) - 2*outer(EA[,l], u) * Eres)
    return (res)
  }
  df.u <- function(u) {
    res <- outer(EA2[,l], u) - sweep(Eres, 1, EA[,l], FUN='*') 
    return (colSums(res))
  }
  Eres <- V - EA %*% U + outer(EA[,l], U[l,])
  u0 <- U[l,]
  u.hat <- optim(u0, f.u, gr=df.u, method='L-BFGS-B')$par
  return (u.hat)
}

update.gamma <- function(EA, EA2) {
  EV <- EA %*% U
  EV2 <- EA2 %*% (U^2) + EV^2 - (EA^2) %*% (U^2)
  res <- 1 / colSums(V^2 - 2 * V * EV + EV2)
  return (res)
}

update.alpha <- function(EA, ElogA) {
  f.eta <- function(eta){
    tmp1 <- exp(eta) * eta - lgamma(exp(eta))
    tmp2 <- sweep(ElogA, 2, exp(eta) - 1, FUN='*') - sweep(EA, 2, exp(eta), FUN='*')
    return (-n_time * sum(tmp1) + sum(tmp2))
  }
  df.eta <- function(eta) {
    res <- -exp(eta) * (n_time * (eta + 1 - digamma(exp(eta))) + colSums(ElogA - EA))
    return (res)
  }
  eta0 <- log(alpha)
  eta.hat <- optim(eta0, f.eta, gr=df.eta, method='L-BFGS-B')$par
  return (exp(eta.hat))
}

comp.obj <- function(res.exp, U, alpha, gamma) {
  EA <- res.exp$EA
  EA2 <- res.exp$EA2
  ElogA <- res.exp$ElogA
  
  EV <- EA %*% U
  EV2 <- EA2 %*% (U^2) + EV^2 - (EA^2) %*% (U^2)
  
  obj <- 0.5 * (n_time * sum(log(gamma)) - sum(sweep((V^2 - 2 * V * EV + EV2), 2, gamma, FUN='*')))
  obj <- obj + n_time * sum(alpha * log(alpha) - lgamma(alpha))
  obj <- obj + sum(sweep(ElogA, 2, alpha-1, FUN='*') - sweep(EA, 2, alpha, FUN='*'))
  return (obj)
}




data <- gen.data(seed)
V <- data$V

maxiter <- 1
threshold <- 0.01
fit <- NULL
stanfile <- 'posterior_approx.stan'
old.obj <- -Inf

seed <- 98765
smoothness <- 100
set.seed(seed)
U <- matrix(rnorm(L * n_freq), nrow=L, ncol=n_freq)
alpha <- rgamma(L, shape=smoothness, rate=smoothness)
gamma <- rgamma(n_freq, shape=smoothness, rate=2*smoothness)

for (i in 1:maxiter) {
   fit <- e.step(stanfile, U, alpha, gamma, model=fit)
   res.exp <- comp.exp(fit)
   res.m <- m.step(res.exp)
   U <- res.m$U
   alpha <- res.m$alpha
   gamma <- res.m$gamma
   obj <- comp.obj(res.exp, U, alpha, gamma)
   improvement <- (obj - old.obj) / abs(obj)
   print(paste('After Iteration', i, 'Improvement:', improvement))
   if (improvement < threshold) {
     break
   }
   old.obj <- obj
}

