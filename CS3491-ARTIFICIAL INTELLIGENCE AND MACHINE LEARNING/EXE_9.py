import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Set the number of clusters (for Iris dataset, usually k=3 is appropriate)
k = 3

# Initialize and fit the KMeans clustering model
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Predict the cluster labels for the dataset
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

# Print clustering results
print("Cluster Labels:", labels)
print("Cluster Centers:\n", centroids)
print("Inertia:", inertia)

# (Optional) Visualize clusters using the first two features
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering on Iris Dataset')
plt.legend()
plt.show()
