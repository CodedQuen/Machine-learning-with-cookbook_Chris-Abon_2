# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create logistic regression
logistic = linear_model.LogisticRegression()

# Create range of 20 candidate values for C
C = np.logspace(0, 4, 20)

# Create hyperparameter options
hyperparameters = dict(C=C)
# Create grid search

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)

# Conduct nested cross-validation and outut the average score
cross_val_score(gridsearch, features, target).mean()

gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1)

best_model = gridsearch.fit(features, target)

scores = cross_val_score(gridsearch, features, target)
