import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# Read the data from the CSV file
data = pd.read_csv('data.csv')
scaler = MinMaxScaler()

# Prepare the data
X = data[['peak', 'ds']].values
y_mass = data['mass'].values / 1e30

scaler.fit(X)
X_scaled = scaler.transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_mass, test_size=0.2, random_state=42)

# Create and train the MLPRegressor model
model = MLPRegressor(hidden_layer_sizes=(100,), random_state=1, max_iter=500)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Print the predicted mass values for the test set
#for i, y_pred_i in enumerate(y_pred):
    #print(f"Predicted mass for test data point {i+1}: {y_pred_i}, {y_mass[i]}")

# Calculate the percentage difference
percentage_diff = (y_pred - y_test) / y_test * 100

# Convert negative percentages to positive
percentage_diff = np.abs(percentage_diff)

# Print the predicted mass values and percentage difference for the test set
"""for i, (y_pred_i, percentage_diff_i) in enumerate(zip(y_pred, percentage_diff)):
    print(f"Test data point {i+1}: Predicted mass = {y_pred_i}, Percentage Difference = {percentage_diff_i:.2f}%")"""

# Calculate and display the average percentage difference
avg_percentage_diff = np.mean(percentage_diff)
print(f"Mass Average Percentage Difference: {avg_percentage_diff:.2f}%")





y_energy= data['energy'].values / 1e43 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_energy, test_size=0.2, random_state=42)

model = MLPRegressor(hidden_layer_sizes=(100,), random_state=1, max_iter=500)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


# Print the predicted mass values for the test set
#for i, y_pred_i in enumerate(y_pred):
    #print(f"Predicted mass for test data point {i+1}: {y_pred_i}, {y_mass[i]}")

# Calculate the percentage difference
percentage_diff = (y_pred - y_test) / y_test * 100

# Convert negative percentages to positive
percentage_diff = np.abs(percentage_diff)

# Calculate and display the average percentage difference
avg_percentage_diff = np.mean(percentage_diff)
print(f"Energy Average Percentage Difference: {avg_percentage_diff:.2f}%")
