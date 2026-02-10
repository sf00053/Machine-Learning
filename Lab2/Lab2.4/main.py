# Import packages and functions
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Input standardized feature values for a sample instance
wing = float(input())
weight = float(input())
culmen = float(input())

# Load the Hawks dataset
hawks = pd.read_csv("hawks.csv")

# Define input features and output features
X = hawks[["Wing", "Weight", "Culmen"]]
y = hawks[["Species"]]

# Standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize a linear discriminant model
linearModel = LinearDiscriminantAnalysis(n_components = 2)

# Fit the model
linearModel.fit(X, np.ravel(y))

# Discriminant intercepts
intercept = linearModel.intercept_
print(intercept)

# Discriminant weights
weights = linearModel.coef_
print(weights)

# Calculate prediction
XNew = np.array([[wing, weight,culmen]])
preds = linearModel.predict(XNew)
print("Predicted species is ", preds)