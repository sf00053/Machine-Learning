import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Silence warning from sklearn
import warnings
warnings.simplefilter("ignore")

# Input the random state
rand = int(input())

# Load sample set by a user-defined random state into a dataframe. 
NBA = pd.read_csv("nbaallelo_log.csv").sample(n=500, random_state=rand)

# Create binary feature for game_result with 0 for L and 1 for W
NBA["win"] = NBA["game_result"].replace(to_replace = ["L","W"], value = [int(0), int(1)])

# Store relevant columns as variables
X = NBA[["elo_i"]]
y = NBA[["win"]]

# Build logistic model with default parameters, fit to X and y
logModel = LogisticRegression()
logModel.fit(X,np.ravel(y))

# Use the model to predict the classification of instances in X
logPredY = logModel.predict(X)

# Calculate the confusion matrix for the model
confMatrix = metrics.confusion_matrix(np.ravel(y), logPredY)
print("Confusion matrix:\n", confMatrix)

# Calculate the accuracy for the model
accuracy = metrics.accuracy_score(np.ravel(y), logPredY)
print("Accuracy:", round(accuracy,3))

# Calculate the precision for the model
precision = metrics.precision_score(np.ravel(y), logPredY)
print("Precision:", round(precision,3))

# Calculate the recall for the model
recall = metrics.recall_score(np.ravel(y), logPredY)
print("Recall:", round(recall, 3))

# Calculate kappa for the model
kappa = metrics.cohen_kappa_score(np.ravel(y), logPredY)
print("Kappa:", round(kappa, 3))