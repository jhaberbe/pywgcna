import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn.cluster
from sklearn.metrics import calinski_harabasz_score

def spectral_clustering_method(TOM: np.array) -> np.array:
    spectral_clustering = sklearn.cluster.SpectralClustering(n_clusters=8, affinity="precomputed", assign_labels="discretize", random_state=42069)

    best_score = -1
    best_n_clusters = None
    best_labels = None

    for n_clusters in tqdm(range(2, 21)):  # Iterating over possible cluster counts
        spectral_clustering.set_params(n_clusters=n_clusters)
        labels = spectral_clustering.fit_predict(TOM)
        score = calinski_harabasz_score(TOM, labels)
        print(n_clusters, score)
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels

    print(f"Optimal number of clusters: {best_n_clusters} with Calinski-Harabasz score: {best_score}")

    return best_labels