if (!require('deSolve')) {
  install.packages('deSolve')
}
library(deSolve)
a <- 0.3; beta <- 0.1
init <- c(S=0.99, I=0.01, R=0)

sir <- function(t, state, parms) {
  with(as.list(c(state, parms)), {
    dS <- -a * S * I
    dI <- a * S * I - beta * I
    dR <- beta * I
    list(c(dS, dI, dR))
  })
}

times <- seq(0, 160, by=0.1)
out <- ode(y=init, times=times, func=sir, parms=c(a=a, beta=beta))
plot(out[, 'time'], out[, 'I'], type='l', xlab='Zeit', ylab='I(t)')