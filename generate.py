import math
import numpy as np
import sympy as sp
from scipy import integrate
import pickle

# Step 1: Calculate x,y

mass_start = 0.8 * 2.0e33   # Starting mass
mass_stop = 1.4 * 2.0e33    # Ending mass
mass_step = 0.1 * 2.0e33    # Mass increment

masses = np.arange(mass_start, mass_stop, mass_step)  # List of masses

energy_start = 10**43       # Staring energy
energy_end = 10**44         # Ending energy
energy_step = 3*(10**43)    # Energy increment

energy = np.arange(energy_start,  energy_end, energy_step)  # List of Energy

time_start = 0.5   # Starting time
time_stop = 40     # Ending time
time_step = 0.5    # Time increment

time = np.arange(time_start, time_stop, time_step)     # List of time


x = []
y = []


nickel = 7.605*(10**5)  # Nickel
b = 13.7
k = 0.02


for m in masses:
    to = (k * m) / b
    for e in energy:
        vsc = math.sqrt(2 * e / m)
        th = 1 / vsc
        tm = math.sqrt(2 * (to * th))
        for t in time:
            t = t * 24 * 60 * 60  # convert days to seconds
            x.append(t / (tm))
            y.append(tm / (2 * nickel))

# Step 2: Integrate


def f(z, y):
    return np.exp(-2 * z * y + z**2) * 2 * z  # Define function


final = []
for x_val, y_val in zip(x, y):
    result, err = integrate.quad(f, 0, x_val, args=(y_val,))
    result *= np.exp(-x_val**2)  # multiply by e^-x^2
    final.append(result)

# Step 3: Output Values to Pickle
with open('results.pickle', 'wb') as file:
    pickle.dump(final, file)
