import pickle

# Step 1: Load the pickle file
with open('results_with_time.pickle', 'rb') as file:
    data = pickle.load(file)

# Step 2: Print the time and result pairs
for time_original, result in data.items():
    print(f"Time: {time_original}, Result: {result}")
