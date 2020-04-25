# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# Set random seed
np.random.seed(0)
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])
# Create dictionary with candidate learning algorithms and their hyperparameters
search_space = [{"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l1', 'l2'],
                 "classifier__C": np.logspace(0, 4, 10)},
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_features": [1, 2, 3]}]
# Create grid search
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)
# Fit grid search
best_model = gridsearch.fit(features, target)

# View best model
best_model.best_estimator_.get_params()["classifier"]

# Predict target vector
best_model.predict(features)
