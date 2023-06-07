import pickle
import numpy as np

# Load the pickle file
with open('results.pickle', 'rb') as file:
    data = pickle.load(file)

# View the loaded data keys with rounded values in scientific notation
for key, value in data.items():
    # Format each value in the key tuple using scientific notation
    rounded_key = tuple(f"{val:.2e}" for val in key)
    # Format the output value using scientific notation
    rounded_value = f"{value:.2e}"
    print(f"{rounded_key}: {rounded_value}")
