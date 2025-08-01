if (!require('deSolve')) {
  install.packages('deSolve')
}
library(deSolve)
kA <- 0.01; kB <- 0.015
r1 <- 0.0; r2 <- 1.2
b1 <- 0.005; b2 <- 0.07
init <- c(A=100, B=100)

lanchester <- function(t, state, parms) {
  with(as.list(c(state, parms)), {
    dA <- -kB * B + r1 - b1 * A
    dB <- -kA * A + r2 - b2 * B
    list(c(dA, dB))
  })
}

times <- seq(0, 500, by=0.1)
out <- ode(y=init, times=times, func=lanchester,
           parms=c(kA=kA, kB=kB, r1=r1, r2=r2, b1=b1, b2=b2))
plot(out[, 'time'], out[, 'A'], type='l', xlab='Zeit', ylab='Flugzeige')
lines(out[, 'time'], out[, 'B'], col='red')