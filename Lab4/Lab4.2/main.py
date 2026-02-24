import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.linear_model import LinearRegression

# Input the random state
rand = int(input())

# Load sample set by a user-defined random state into a dataframe
diamonds = pd.read_csv("diamonds.csv").sample(n=500, random_state=rand)

# Define input and output features
X = diamonds[["carat", "table"]]
y = diamonds["price"]

# Initialize and fit a multiple linear regression model
linear_model = LinearRegression()
linear_model.fit(X, np.ravel(y))

# Use the model to predict the classification of instances in X
mlrPredY = linear_model.predict(X)

# Calculate mean absolute error for the model
mae = metrics.mean_absolute_error(np.ravel(y), mlrPredY)
print("MAE:", round(mae, 3))

# Calculate mean squared error for the model
mse = metrics.mean_squared_error(np.ravel(y), mlrPredY)
print("MSE:", round(mse, 3))

# Calculate root mean squared error for the model
rmse = metrics.root_mean_squared_error(np.ravel(y), mlrPredY)
print("RMSE:", round(rmse, 3))

# Calculate R-squared for the model
r2 = metrics.r2_score(np.ravel(y), mlrPredY)
print("R-squared:", round(r2, 3))