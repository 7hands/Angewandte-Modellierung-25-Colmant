import numpy as np
import matplotlib.pyplot as plt

# Einzelne Ziehung von 6 Zahlen
zahlen = np.random.choice(np.arange(1,50), size=6, replace=False)
print("Gezogene Zahlen:", np.sort(zahlen))

# 100 Ziehungen und Histogramm
n_draws = 100
# Für jede Ziehung einzeln ohne replacement
draws = np.array([np.random.choice(np.arange(1,50), size=6, replace=False) for _ in range(n_draws)])
numbers = draws.flatten()

plt.hist(numbers, bins=np.arange(1,51)-0.5, edgecolor='black')
plt.xlabel('Zahl')
plt.ylabel('Häufigkeit')
plt.title(f'Histogramm der {n_draws} Lotterie-Ziehungen')
plt.xticks(np.arange(1,50,2))
plt.show()