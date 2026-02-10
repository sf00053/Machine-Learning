# Import needed packages for regression
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Silence warning from sklearn
import warnings
warnings.filterwarnings("ignore")

# Input feature values for a sample instance
carat = float(input())
table = float(input())

diamonds = pd.read_csv("diamonds.csv")

# Define input and output features
X = diamonds[["carat", "table"]]
y = diamonds["price"]

# Initialize a multiple linear regression model
multLinearModel = LinearRegression()

# Fit the multiple linear regression model to the input and output features
multLinearModel.fit(X,y)

# Get estimated intercept weight
intercept = multLinearModel.intercept_
print("Intercept is", round(intercept, 3))

# Get estimated weights for carat and table features
coefficients = multLinearModel.coef_
print("Weights for carat and table features are", np.round(coefficients, 3))

# Predict the price of a diamond with the user-input carat and table values
prediction = multLinearModel.predict([[carat, table]])
print("Predicted price is", np.round(prediction, 2))