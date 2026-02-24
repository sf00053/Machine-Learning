import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.inspection import PartialDependenceDisplay
from sklearn import metrics

# Load diamonds sample into dataframe
diamonds = pd.read_csv("diamonds.csv").sample(n=50, random_state=42)

# Get user-input features
feature1 = input()
feature2 = input()

# Define input and output features
X = diamonds[[feature1, feature2]]
y = diamonds["price"]

# Initialize and fit a multiple linear regression model
lModel = LinearRegression()
lModel.fit(X, np.ravel(y))

# Use the model to predict the classification of instances in X
mlrPredY = lModel.predict(X)

# Compute prediction errors
mlrPredError = y- mlrPredY

# Plot prediction errors vs predicted values. Label the x-axis as 'Predicted' and the y-axis as 'Prediction error'
fig = plt.figure()
plt.scatter(mlrPredY, mlrPredError)
plt.xlabel("Predicted")
plt.ylabel("Prediction error")

# Add dashed line at y=0
plt.axhline(0, linestyle="--")

plt.savefig("predictionError.png")

# Generate a partial dependence display for both input features
fig = plt.figure()
PartialDependenceDisplay.from_estimator(lModel, X, [feature1, feature2])

plt.savefig("partialDependence.png")

# Calculate mean absolute error for the model
mae = metrics.mean_absolute_error(y, mlrPredY)
print("MAE:", round(mae, 3))