


?dnorm
?rnorm
?dbinom
?dpois


?mtext



require(graphics)


par(mfrow = c(2,1))
plot(function(x) dnorm(x), -5, 5, tck = 1, ylim = c(0, 0.8), main = "Normal distribution P(x)",
     ylab = "", col.main = "red", cex.main = 2)
curve(dnorm(x), add = TRUE, col = "red", lwd = 4)
curve(dnorm(x, 0, 2.0), add = TRUE, col = "green", lwd = 4)
curve(dnorm(x, 0, 0.5), add = TRUE, col = "blue", lwd = 4)
mtext("green: sd=2", col = "green", line = -6, adj = 0.1, cex = 1.5)
mtext("red: sd=1", col = "red", line = -4, adj = 0, cex = 1.5)
mtext("blue: sd=0.5", col = "blue", line= -8,adj = 0, cex = 1.5)


?dbinom

require(graphics)
# Compute P(45 < X < 55) for X Binomial(100,0.5)
sum(dbinom(46:54, 100, 0.5))

## Using "log = TRUE" for an extended range :
n <- 50
k <- seq(0, n, by = 1)
plot (k, dbinom(k, n, 0.5), type = "p", ylab = "Density", ylim = c(0, 0.16),
      main = "Binomial distribution, n=50", cex.main = 2)
lines(k, dbinom(k, n, 0.5), col = "red", lwd = 3)
lines(k, dbinom(k, n, 0.3), col = "green", lwd = 3)
lines(k, dbinom(k, n, 0.7), col = "blue", lwd = 3)
## extreme points are omitted since dbinom gives 0.
mtext("p=0.5", col = "red", adj = 0)
mtext("n=50", adj = 0.2, line = -2, font = 4, cex = 1.5)
mtext("p=0.3", col = "green", adj = 0.5)
mtext("p=0.7", col = "blue", adj = 1)



n <- 10
k <- seq(0, n, by = 1)
plot (k, dbinom(k, n, 0.5), type = "p", ylab = "Density", ylim = c(0, 0.3),
      main = "Binomial distribution", cex.main = 2)
lines(k, dbinom(k, n, 0.5), col = "red", lwd = 3)
## extreme points are omitted since dbinom gives 0.
mtext("", col = "red", adj = 0)
mtext("n=10, p=0.5", adj = 0.2, line = -2, font = 4, cex = 1.5)
mtext("", col = "green", adj = 0.5)
mtext("", col = "blue", adj = 1)



?dpois

par(mfrow = c(1, 1))


x <- seq(0, 20, 1)
plot(x, dpois(x, 1), type = "l", col = "blue", ylab = "P(n)", xlab = "n", main = "Poisson distribution", lwd = 3, cex.main = 2)
lines(x, dpois(x,5), col = "red", lwd = 3)
lines(x, dpois(x,10), col = "green", lwd = 3)
mtext("a = 5", col = "red", adj = 0.7, line = -4, cex = 1.5)
mtext("a = 10", col = "green", adj = 0.7, line = -6, cex = 1.5)
mtext("a = 1", col = "blue", adj = 0.7, line = -2, cex = 1.5)


RNGkind()
# To generate n pseudo-random numbers in R, use runif(n)
set.seed(666)
runif(2)

RNG.state <- .Random.seed
runif(2)

set.seed(666)
runif(4)
.Random.seed <- RNG.state
runif(2)

f <- runif(200)
plot(f, xlab = "x", ylab = "Random")
hist(f)

nor <- rnorm(100000)
hist(nor)


c5 <- 0
for (i in 1:100000){
  if (nor[i]>5)
    c5 = c5 + 1
}

c5

head(nor)


c3 <- 0
for (i in 1:100000){
  if (nor[i]>3)
    c3 = c3 + 1
}

c3


c4 <- 0
for (i in 1:100000){
  if (nor[i]>4)
    c4 = c4 + 1
}

c4
########################################################
# Clear workspace
rm(list = ls())



