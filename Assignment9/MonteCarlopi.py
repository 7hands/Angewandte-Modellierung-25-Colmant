import random
import math

def estimate_pi(n):
    count = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if (x - 0.5)**2 + (y - 0.5)**2 <= 0.5**2:
            count += 1
    return 4 * count / n

def estimate_beta(z, w, n):
    total = 0.0
    for _ in range(n):
        x = random.random()
        total += x**(z - 1) * (1 - x)**(w - 1)
    return total / n

# Number of samples
N = 100_000_000

# Exercise 4: Monte Carlo π
pi_estimate = estimate_pi(N)

# Exercise 5: Monte Carlo Beta(0.5, 2)
beta_estimate = estimate_beta(0.5, 2, N)
beta_exact = math.gamma(0.5) * math.gamma(2) / math.gamma(2.5)

print(f"Estimated pi (N={N}): {pi_estimate}")
print(f"Estimated B(0.5, 2) (N={N}): {beta_estimate}")
print(f"Exact B(0.5, 2): {beta_exact}")