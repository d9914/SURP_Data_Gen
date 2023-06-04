import pickle

# Load the pickle file
with open('results.pickle', 'rb') as file:
    data = pickle.load(file)

# View the loaded data
print(data)
