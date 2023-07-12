import math
import numpy as np
from scipy import integrate
import pickle
import csv

# Step 1: Calculate x, y

mass_start = 0.8 * 2.0e30   # Starting mass
mass_stop = 1.4 * 2.0e30    # Ending mass
mass_step = 0.01 * 2.0e30    # Mass increment

masses = np.arange(mass_start, mass_stop, mass_step)  # List of masses

energy_start = 10**43       # Staring energy
energy_end = 10**44         # Ending energy
energy_step = .5*(10**43)    # Energy increment

energy = np.arange(energy_start,  energy_end, energy_step)  # List of Energy

time_start = 0.1   # Starting time
time_stop = 40     # Ending time
time_step = 0.1    # Time increment

time = np.arange(time_start, time_stop, time_step)     # List of time


x = []
y = []


nickel = 7.605*(10**5)  # Nickel
b = 13.7        # 
k = 0.02      # Kinetic Energy 
c = 3 * 10**8    # Speed of Light

# Dictionary to store the calculated values with their corresponding (x, y) pairs, data_graph stores result and days
data = {}
data_2={}

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

            if (m, e) not in data:
                data[(m, e)] = []

            result = integrate.quad(
                lambda z, y: np.exp(-2 * z * y + z ** 2) * 2 * z, 0, x_val, args=(y_val,)
            )[0]
            result *= np.exp(-x_val ** 2)  # multiply by e^-x^2
            data[(m, e)].append(result)
           
            if (m, e) not in data_2:
                data_2[(m, e)] = []
            data_2[(m, e)].append(time_original)

        peak=max(data[(m,e)])
        #for d, l in zip(data_2[(m,e)], data[(m,e)]):
            

        ds=data_2[(m,e)]
        ls=data[(m,e)]
        
        for i in range(len(ds)-1):
            if ls[i]==peak:
                peak_day=ds[i]
            if ls[i+1]<peak/2 and ls[i]>peak/2:
                peak_day_low=(ds[i]+ds[i+1])/2

        
        ds=peak_day_low-peak_day
        data[(m,e)]=(peak, ds)
      

# Step 3: Output Values to Pickle
#with open('results_with_time.pickle', 'wb') as file:
    #pickle.dump(data_graph, file)


#with open('results.pickle', 'wb') as file:
    #pickle.dump(data, file)

csv_filename = 'data.csv'

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['mass', 'energy', 'peak', 'ds'])  # Write header row
    for k, v in data.items():
        writer.writerow([k[0], float(k[1]), v[0], v[1]])

print(f"Data has been saved to {csv_filename}")