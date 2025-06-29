# Anzahl der Simulationen
N <- 1e6

# Übung 4: Monte-Carlo-Schätzung von π
estimate_pi <- function(n) {
  x <- runif(n)
  y <- runif(n)
  inside <- (x - 0.5)^2 + (y - 0.5)^2 <= 0.5^2
  return(mean(inside) * 4)
}

# Übung 5: Monte-Carlo-Integral für die Beta-Funktion
estimate_beta <- function(z, w, n) {
  x <- runif(n)
  return(mean(x^(z - 1) * (1 - x)^(w - 1)))
}

# Schätzungen
pi_estimate    <- estimate_pi(N)
beta_estimate  <- estimate_beta(0.5, 2, N)
beta_exact     <- beta(0.5, 2)  # eingebaute Beta-Funktion in R

# Ausgabe
cat("Geschätztes π (N=", N, "): ", pi_estimate, "\n", sep = "")
cat("Geschätztes B(0.5,2) (N=", N, "): ", beta_estimate, "\n", sep = "")
cat("Exaktes B(0.5,2): ", beta_exact, "\n", sep = "")