import numpy as np

zahlen = np.random.choice(np.arange(1,50), size=6, replace=False)
print("Gezogene Zahlen:", np.sort(zahlen))