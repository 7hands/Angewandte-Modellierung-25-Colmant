import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Stichprobe aus einer Standardnormalverteilung
n = 10**5
samples = np.random.normal(loc=0, scale=1, size=n)
# Anzahl der Ereignisse mit |x| > 5
count = np.sum(np.abs(samples) > 5)
# Theoretische Wahrscheinlichkeit und erwartete Anzahl
p_tail = 2 * (1 - norm.cdf(5))
expected = n * p_tail
print(f"Anzahl der Punkte mit |x| > 5: {count}")
print(f"Theoretische Wahrscheinlichkeit für |x|>5: {p_tail:.2e}")
print(f"Erwartete Anzahl bei n={n}: {expected:.2f}")

# Histogramm der Stichprobe mit Markierung bei ±5
plt.hist(samples, bins=100, density=True)
plt.axvline(5, linestyle='--', label='+5')
plt.axvline(-5, linestyle='--', label='-5')
plt.xlabel('Wert')
plt.ylabel('Dichte')
plt.title('Histogramm der Normalverteilung mit 5-Grenzen')
plt.legend()
plt.show()