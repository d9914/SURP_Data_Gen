import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor #LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Read the data from the CSV file
data = pd.read_csv('data.csv')
scaler= MinMaxScaler()

# Prepare the data
X = data[['peak', 'ds']].values
y_mass = data['mass'].values / 1e30

scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_mass, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = MLPRegressor(hidden_layer_sizes=(15,), random_state=1, max_iter=500)
model.fit(X_train, y_train)



# Make predictions on the test set
y_pred = model.predict(X_test)

#Print the predicted x_val values for the test set
for i, y_pred_i in enumerate(y_pred):
    print(f"Predicted x_val for test data point {i+1}: {y_pred_i}, {y_mass[i]}")

# Calculate the percentage difference
percentage_diff = (y_pred - y_test) / y_test * 100

# Print the predicted x_val values and percentage difference for the test set
for i, (y_pred_i, percentage_diff_i) in enumerate(zip(y_pred, percentage_diff)):
    print(f"Test data point {i+1}: Predicted x_val = {y_pred_i}, Percentage Difference = {percentage_diff_i:.2f}%")
