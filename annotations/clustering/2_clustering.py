import numpy as np

embeddings = np.load("embs.npz")['arr_0']
print(embeddings)

import numpy as np
import cupy as cp
from cuml.cluster import DBSCAN
from cuml.decomposition import PCA
from cuml.manifold import UMAP  # RAPIDS UMAP

from cuml.cluster import KMeans
import matplotlib.pyplot as plt

from cuml.cluster import AgglomerativeClustering


# Move embeddings to GPU as a CuPy array
embeddings_cp = cp.asarray(embeddings)

print("PCA")
# PCA (GPU-accelerated)
# pca_components = 2
# pca = PCA(n_components=pca_components)
# reduced_embeddings_cp = pca.fit_transform(embeddings_cp)
pca = UMAP(n_components=20, n_neighbors=30, min_dist=0, random_state=42)
reduced_embeddings_cp = pca.fit_transform(embeddings)


print("KMEANS")
# K-Means (GPU-accelerated)
# num_clusters = 50
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# labels_cp = kmeans.fit_predict(reduced_embeddings_cp)

num_clusters = 50
kmeans = AgglomerativeClustering(n_clusters=num_clusters)
labels_cp = kmeans.fit_predict(reduced_embeddings_cp)

# eps = 0.1
# min_samples = 5
# dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
# labels_cp = dbscan.fit_predict(reduced_embeddings_cp)


print("CPU")
# Convert results back to CPU (if needed for visualization)
reduced_embeddings = reduced_embeddings_cp#.get()
labels = labels_cp#.get()
# cluster_centers = kmeans.cluster_centers_#.get()

cluster_labels = np.unique(labels[labels != -1])
closest_indices = []
cluster_centers = []

for cluster in cluster_labels:
    # Get the indices of points in the current cluster
    cluster_indices = np.where(labels == cluster)[0]
    
    # Calculate the centroid of the cluster
    cluster_points = reduced_embeddings[cluster_indices]
    cluster_center = cluster_points.mean(axis=0)
    cluster_centers.append(cluster_center)

cluster_centers = np.array(cluster_centers)
print(cluster_centers.shape)

# Print cluster centers and explained variance ratio
print("Cluster centers (in PCA-reduced space):")
print(cluster_centers)
# print("Explained variance ratio by PCA components:")
# print(pca.explained_variance_ratio_)

import joblib
import numpy as np

# Assuming you've run PCA and KMeans as in the previous example,
# and have the following variables:
# pca, reduced_embeddings_cp, kmeans, labels_cp


print("Saving")
# Save PCA Results
# PCA components, explained variance, and mean can be important for later transformation of new data.
# np.save("pca_components.npy", pca.components_.get())
# np.save("pca_explained_variance.npy", pca.explained_variance_ratio_.get())
# np.save("pca_mean.npy", pca.mean_.get())

# Save transformed embeddings after PCA
np.save("reduced_embeddings.npy", reduced_embeddings)

# Save KMeans results
np.save("kmeans_cluster_centers.npy", cluster_centers)
np.save("kmeans_labels.npy", labels)

# Optionally, save the entire PCA and KMeans models using joblib or pickle
# Note: Make sure your environment is consistent when loading these back.
# GPU-accelerated models may need the same RAPIDS environment to load properly.

joblib.dump(pca, "pca_model.joblib")
# joblib.dump(kmeans, "kmeans_model.joblib")

print("All results saved successfully.")

closest_indices = []

# For each cluster center, find the closest data point
for i, center in enumerate(cluster_centers):
    # Compute distances from all points to this cluster center
    distances = np.linalg.norm(reduced_embeddings - center, axis=1)
    
    # Find the index of the closest point
    closest_idx = np.argmin(distances)
    closest_indices.append(closest_idx)

# Save the closest indices to a file for later use if needed
np.save("closest_indices_per_cluster.npy", np.array(closest_indices))

# Lookup the corresponding string for each closest index
import pandas as pd
distinct_titles = set()
data_file = 'outcomes.csv' 
df = pd.read_csv(data_file)
distinct_titles = list(set(df['object'].values.tolist()))

my_strings = distinct_titles
for cluster_idx, point_idx in enumerate(closest_indices):
    corresponding_str = my_strings[point_idx]
    print(f"Cluster {cluster_idx}: Closest point index = {point_idx}, Corresponding str = {corresponding_str}")

# # Visualization (only if pca_components=2)
# if pca_components == 2:
if True:
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.002)
    plt.savefig("dots.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.2)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                c='red', s=200, alpha=0.75, label='Centroids')
    plt.title("PCA-Reduced Embeddings with K-Means Clusters (GPU-Accelerated)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    # plt.show()
    plt.savefig("clusters.png")

selected_samples = {}
import random
random.seed(0)
# Get unique cluster labels (ignore noise points labeled -1)
cluster_labels = np.unique(labels[labels != -1])

# Randomly select 5 samples from each cluster
for cluster in cluster_labels:
    # Get the indices of points in the current cluster
    cluster_indices = np.where(labels == cluster)[0]
    
    # Select up to 5 random indices from the cluster
    selected_indices = random.sample(list(cluster_indices), min(5, len(cluster_indices)))
    
    # Store the selected indices and their corresponding text
    selected_samples[cluster] = [(idx, my_strings[idx]) for idx in selected_indices]

# Print selected samples
for cluster, samples in selected_samples.items():
    print(f"Cluster {cluster}:")
    for idx, text in samples:
        print(f"  Index: {idx}, Text: {text}")

# Optional: Save results to a file
np.save("selected_samples_per_cluster.npy", selected_samples)


# take many subpoints in each cluster and add gpt to come up with a category
