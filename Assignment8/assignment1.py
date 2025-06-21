import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parameter
a, beta = 0.3, 0.1
S0, I0, R0 = 0.99, 0.01, 0.0

def sir(y, t, a, beta):
    S, I, R = y
    dS = -a * S * I
    dI = a * S * I - beta * I
    dR = beta * I
    return [dS, dI, dR]

t = np.linspace(0, 160, 1600)
sol = odeint(sir, [S0, I0, R0], t, args=(a, beta))

plt.plot(t, sol[:, 1], label='I(t)')
plt.xlabel('Zeit')
plt.ylabel('Anteil infiziert')
plt.legend();
plt.show()