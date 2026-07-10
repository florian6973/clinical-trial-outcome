import numpy as np
from sklearn.decomposition import PCA
from hdbscan import flat

mm = np.memmap("array-outcome.dat", mode='r', shape=(438026, 4096), dtype='float16')

pca = PCA(n_components=20)
print("Fit PCA")
features = pca.fit_transform(mm)


# print("Cluster")
# clusterer = flat.HDBSCAN_flat(features, 100, prediction_data=True, metric='euclidean') #algorithm='generic', metric='cosine') #metric='euclidean') #, cluster_selection_method='leaf')
# # https://github.com/scikit-learn-contrib/hdbscan/issues/69
# # https://github.com/scikit-learn-contrib/hdbscan/issues/345
# # https://github.com/scikit-learn-contrib/hdbscan/issues/291

# print("Save labels")
# np.savez('cluster-labels.npz', labels=clusterer.labels_)
# print("End")


print("Clustering")
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=100)

# Fit the model to the data
kmeans.fit(features)

print("Saving")
# Get the centroids and labels (which cluster each point belongs to)
# https://arxiv.org/html/2402.14526v1
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
# print(labels)
np.savetxt("labels-all.txt", labels)
np.savetxt("centroids-all.txt", centroids)

labels = np.loadtxt("labels-all.txt")
print(np.unique(labels))

unique_labels = np.unique(labels)

# For each unique label, randomly select one occurrence
selected_labels = np.array([np.random.choice(np.where(labels == label)[0]) for label in unique_labels])

# Print the selected indices and their corresponding label values
print("Selected indices:", selected_labels)
print("Corresponding label values:", labels[selected_labels])

np.savetxt("selected-idxes.txt", selected_labels)