import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

taxis = pd.read_csv("taxis.csv")

# Create dataframe X with features passengers and distance
X = taxis[['passengers', 'distance']]

# Create dataframe y with feature fare. 
y = taxis['fare']

# Set aside 10% of instances for testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.1 ,random_state = 42)

# Split training again into 70% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = (0.2/0.9), random_state = 42)

# Initialize a linear regression model
linRegModel = LinearRegression()

# Fit the model with 15-fold cross-validation to the training data, 
# using "explained_variance" as the performance metric
cv_results = cross_validate(linRegModel, X_train,y_train,cv =15, scoring = "explained_variance")

# Print the explained variance for each fold
print("Test score:",cv_results["test_score"])