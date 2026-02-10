# Import the necessary modules
import numpy as np
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
# Load the dataset
skySurvey = pd.read_csv("SDSS.csv")

# Get user defined random seed
seed = int(input())
np.random.seed(seed)
ssSample = skySurvey.sample(n=500)

# Create a new feature from u - g
ssSample["u_g"] = ssSample["u"] - ssSample["g"]

# Create dataframe X with features redshift and u_g
X = ssSample[["redshift", "u_g"]]

# Create dataframe y with feature class
y = ssSample["class"]

# Initialize a Gaussian naive Bayes model
skySurveyNBModel = GaussianNB()

# Fit the model
skySurveyNBModel.fit(X, np.ravel(y))

# Calculate the proportion of instances correctly classified
score = skySurveyNBModel.score(X, y)

# Print accuracy score
print("Accuracy score is ", end="")
print('%.3f' % score)