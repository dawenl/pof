library(R.matlab)

data <- readMat('sa1.mat')
V <- t(data$V)

n_time <- nrow(V)
n_freq <- ncol(V)
L <- 40