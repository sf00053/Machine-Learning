import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

taxis = pd.read_csv("taxis.csv")

# Create dataframe X with the feature distance
X = taxis[['distance']]
# Create dataframe y with the feature fare
y = taxis['fare']

# Set aside 10% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

# Split training again into 80% training and 10% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1/0.9, random_state = 42)


# Initialize a linear regression model
SLRModel = LinearRegression()
# Initialize a k-nearest neighbors regression model with k = 3
knnModel = KNeighborsRegressor(n_neighbors = 3)

# Define a set of 10 cross-validation folds with random_state=42
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)

# Fit k-nearest neighbors with cross-validation to the training data
knnResults = cross_validate(knnModel, X_train, y_train, cv = kf,return_train_score=False)

# Find the test score for each fold
knnScores = knnResults["test_score"]
print("k-nearest neighbor scores:", knnScores.round(3))

# Calculate descriptive statistics for k-nearest neighbor model
print("Mean:", knnScores.mean().round(3))
print("SD:", knnScores.std().round(3))

# Fit simple linear regression with cross-validation to the training data
SLRModelResults = cross_validate(SLRModel, X_train, y_train, cv = kf, return_train_score=False)

# Find the test score for each fold
SLRScores = SLRModelResults["test_score"]
print("Simple linear regression scores:", SLRScores.round(3))

# Calculate descriptive statistics simple linear regression model
print("Mean:", SLRScores.mean().round(3))
print("SD:", SLRScores.std().round(3))