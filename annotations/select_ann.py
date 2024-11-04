source_file = "/nlp/projects/llama/outputs/outcome_embeddings.parquet"

import pandas as pd

# https://stackoverflow.com/questions/48269092/hdbscan-python-choose-number-of-clusters

# embeddings = pd.read_parquet(source_file)
# sub_embeddings= embeddings.iloc[:1000]
# print(sub_embeddings)
# sub_embeddings.to_parquet('sample.parquet')

from sklearn.decomposition import PCA
from hdbscan import flat
import numpy as np

def cluster():
    print("Embeddings")
    embeddings = pd.read_parquet(source_file)
    # embeddings = pd.read_parquet('sample.parquet')
    # print(embeddings['Embedding'].values[0])

    # convert to array
    # print(len(embeddings['Embedding'].values[1].split(',')[0]))

    print("Conversion")
    embeddings = [eval(x) for x in embeddings['Embedding'].values]
    print("Array")
    array = np.array(embeddings)

    np.savez("array.npz", array=array)
    # exit()
    # # pca to 20
    # pca = PCA(n_components=20)
    # features = pca.fit_transform(embeddings['Embedding'].values)
    # clusterer = flat.HDBSCAN_flat(array, 100, prediction_data=True, cluster_selection_method='leaf')
    # print(clusterer.labels_)
    # print(clusterer.labels_)
    print("Clustering")
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=100)

    # Fit the model to the data
    kmeans.fit(array)

    print("Saving")
    # Get the centroids and labels (which cluster each point belongs to)
    # https://arxiv.org/html/2402.14526v1
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    # print(labels)
    np.savetxt("labels-all.txt", labels)
    np.savetxt("centroids-all.txt", centroids)

cluster() # https://info.ornl.gov/sites/publications/files/Pub32719.pdf


# to get representative
# https://medium.com/@megha.natarajan/understanding-the-intuition-behind-cluster-centroids-smote-and-smoteen-techniques-for-dealing-058f3233abeb

# https://medium.com/@konyakinsergey/farthest-point-sampling-for-k-means-clustering-23a6dfc2dfb1
# https://www.qualtrics.com/experience-management/research/representative-samples/#:~:text=Random%20sampling%20is%20a%20method,sample%20of%20a%20whole%20population.

# https://www.scribbr.com/methodology/cluster-sampling/
#https://stackoverflow.com/questions/48269092/hdbscan-python-choose-number-of-clusters 

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9335369/ k-medoid
# https://www.investopedia.com/ask/answers/051815/what-difference-between-systematic-sampling-and-cluster-sampling.asp

# print(embeddings)