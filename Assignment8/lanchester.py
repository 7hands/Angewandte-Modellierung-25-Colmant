import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameter
kA, kB = 0.01, 0.015
r1, r2 = 0.8, 1.2
b1, b2 = 0.005, 0.007
A0, B0 = 150, 100

def lanchester(y, t, kA, kB, r1, r2, b1, b2):
    A, B = y
    dA = -kB * B + r1 - b1 * A
    dB = -kA * A + r2 - b2 * B
    return [dA, dB]

t = np.linspace(0, 50, 500)
sol = odeint(lanchester, [A0, B0], t, args=(kA, kB, r1, r2, b1, b2))

plt.plot(t, sol[:,0], label='A(t)')
plt.plot(t, sol[:,1], label='B(t)')
plt.xlabel('Zeit')
plt.ylabel('Flugzeuge')
plt.legend();
plt.show()
