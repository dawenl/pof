library(R.matlab)

data <- readMat('sa1.mat')
V <- t(data$V)

Time <- nrow(V)
Freq <- ncol(V)
L <- 40
