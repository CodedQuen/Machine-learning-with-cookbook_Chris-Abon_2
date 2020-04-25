# Load libraries
from sklearn import preprocessing
import numpy as np

# Create feature
features = np.array([[-100.1, 3240.1],
                     [-200.2, -234.1],
                     [5000.5, 150.1],
                     [6000.6, -125.1],
                     [9000.9, -673.1]])

# Create scaler
scaler = preprocessing.StandardScaler()

# Transform the feature
features_standardized = scaler.fit_transform(features)

# Show feature
features_standardized

# Print mean and standard deviation
print("Mean:", round(features_standardized[:,0].mean()))
print('"Standard deviation:", features_standardized[:,0].std())
