import math
import numpy as np
import sympy as sp
from scipy import integrate
import pickle

# Step 1: Calculate x, y

mass_start = 0.8 * 2.0e30   # Starting mass
mass_stop = 1.4 * 2.0e30    # Ending mass
mass_step = 0.1 * 2.0e30    # Mass increment

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
b = 13.7        # 
k = 0.02      # Kinetic Energy 
c = 3 * 10**8    # Speed of Light

# Dictionary to store the calculated values with their corresponding (x, y) pairs, data_graph stores result and days
data = {}
data_graph={}

for m in masses:
    to = (k * m) / (b * c)
    for e in energy:
        vsc = math.sqrt(2 * e / m)
        th = 1 / vsc
        tm = math.sqrt(2 * (to * th))
        for t in time:
            time_original = t
            t = t * 24 * 60 * 60  # convert days to seconds
            x_val = t / tm
            y_val = tm / (2 * nickel)

            result = integrate.quad(
                lambda z, y: np.exp(-2 * z * y + z ** 2) * 2 * z, 0, x_val, args=(y_val,)
            )[0]
            result *= np.exp(-x_val ** 2)  # multiply by e^-x^2
            data[(x_val, y_val)] = result
            data_graph[time_original] = result

# Step 3: Output Values to Pickle
with open('results_with_time.pickle', 'wb') as file:
    pickle.dump(data_graph, file)


#with open('results.pickle', 'wb') as file:
    #pickle.dump(data, file)