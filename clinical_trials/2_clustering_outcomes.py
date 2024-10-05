import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import cuml
import os

# Load the embeddings from the CSV file using Dask with tqdm progress bar
print('Loading the embeddings from the CSV file')
embeddings_file = '/nlp/projects/llama/outputs/outcome_embeddings.csv'
data = pd.read_csv(embeddings_file)

# Take a sample of 10,000 rows for testing
print('Taking a sample of 10,000 rows for testing')
data_sample = data.sample(n=10000, random_state=42)

print('Data sample length:', len(data_sample))

# Extract the embeddings and convert them to a numpy array
embeddings = data_sample['Embedding'].apply(lambda x: np.fromstring(x, sep=',')).tolist()
embeddings_array = np.array(embeddings)

# Reduce dimension to 20 using PCA
print('Reducing dimensions to 20 using PCA')
pca = PCA(n_components=20, random_state=42)
embeddings_pca = pca.fit_transform(embeddings_array)

# Change to HDBSCAN clustering
print('Performing HDBSCAN clustering')
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
data_sample['Cluster'] = clusterer.fit_predict(embeddings_pca)

# Reduce to 2 dimensions using UMAP for plotting
print('Reducing dimensions to 2 using UMAP for plotting')
umap_reducer = cuml.UMAP(n_components=2, random_state=42)
embeddings_umap = umap_reducer.fit_transform(embeddings_pca)

# Plot the clusters
print('Plotting the clusters')
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embeddings_umap[:, 0], y=embeddings_umap[:, 1], hue=data_sample['Cluster'], palette='viridis')
plt.title('UMAP Clustering Chart')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(title='Cluster')

# Save the plot as a PNG file in the outputs folder
output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)
plt.savefig(f'{output_folder}/umap_clustering_chart.png')

plt.show()