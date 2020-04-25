# Load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

# Load data with only two features
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

# Create linear regression
regression = LinearRegression()

# Fit the linear regression
model = regression.fit(features, target)

# View the intercept
model.intercept_

# View the feature coefficients
model.coef_

# First value in the target vector multiplied by 1000
target[0]*1000

# Predict the target value of the first observation, multiplied by 1000
model.predict(features)[0]*1000

# First coefficient multiplied by 1000
model.coef_[0]*1000
