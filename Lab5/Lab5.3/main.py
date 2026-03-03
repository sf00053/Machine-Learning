import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

diamonds = pd.read_csv("diamonds.csv")

# Create dataframe X with the features carat and depth
X = diamonds[['carat', 'depth']]
# Create dataframe y with the feature price
y = diamonds['price']

# Create training/testing split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize elastic net model
ENModel = ElasticNet(random_state = 0)

# Create tuning grid
alpha = {'alpha': [0.1, 0.5, 0.9, 1.0]}

# Initialize tuning grid and fit to training data
ENTuning =GridSearchCV(ENModel, alpha, cv =10)
ENTuning.fit(X_train, y_train)

# Mean testing score for each lambda and best model
print("Mean testing scores:", ENTuning.cv_results_['mean_test_score'])
print("Best estimator:", ENTuning.best_estimator_)