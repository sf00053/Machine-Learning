# Import needed packages for regression
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Input standardized feature values for a sample instance
carat = float(input())
table = float(input())

diamonds = pd.read_csv("diamonds.csv")

# Define input and output features
X = diamonds[["carat","table"]]
y = diamonds[["price"]]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize a model using elastic net regression with a regularization strength of 6, and l1_ratio=0.4
eModel = ElasticNet(alpha = 6, l1_ratio = 0.4)

# Fit the elastic net model to the input and output features
eModel.fit(X, y)

# Get estimated intercept weight
intercept = eModel.intercept_
print("Intercept is", np.round(intercept, 3))

# Get estimated weights for carat and table features
coefficients = eModel.coef_
print("Weights for carat and table features are", np.round(coefficients, 3))

# Predict the price of a diamond with the user-input carat and table values
prediction = eModel.predict([[carat, table]])
print("Predicted price is", np.round(prediction, 2))