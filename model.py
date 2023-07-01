import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the data from the CSV file
data = pd.read_csv('results.csv')

# Prepare the data
X = data[['y_val', 'result']].values
y = data['x_val'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

"""Print the predicted x_val values for the test set
for i, y_pred_i in enumerate(y_pred):
    print(f"Predicted x_val for test data point {i+1}: {y_pred_i}")"""

# Calculate the percentage difference
percentage_diff = (y_pred - y_test) / y_test * 100

# Print the predicted x_val values and percentage difference for the test set
for i, (y_pred_i, percentage_diff_i) in enumerate(zip(y_pred, percentage_diff)):
    print(f"Test data point {i+1}: Predicted x_val = {y_pred_i}, Percentage Difference = {percentage_diff_i:.2f}%")

