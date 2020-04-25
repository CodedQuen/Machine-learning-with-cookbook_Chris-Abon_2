# Load libraries
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create logistic regression
logistic = linear_model.LogisticRegression()
# Create range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']
# Create range of candidate regularization hyperparameter values
C = np.logspace(0, 4, 10)
# Create dictionary hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# Fit grid search
best_model = gridsearch.fit(features, target)
np.logspace(0, 4, 10)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# Predict target vector
best_model.predict(features)
