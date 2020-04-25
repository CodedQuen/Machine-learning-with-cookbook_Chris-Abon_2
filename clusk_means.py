# Load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load data
iris = datasets.load_iris()
features = iris.data

# Standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

# Create k-mean object
cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)

# Train model
model = cluster.fit(features_std)

# View predict class
model.labels_

# Predict observation's cluster
model.predict(new_observation)

# View cluster centers
model.cluster_centers_
