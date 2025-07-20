import numpy as np
import hdbscan
from sklearn.decomposition import PCA

class Solver:
    def solve(self, problem, **kwargs):
        dataset = np.array(problem["dataset"])
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        n = len(dataset)
        dim = len(dataset[0]) if n > 0 else 0
        
        # Apply dimensionality reduction for high-dimensional data
        if dim > 50 and n > 100:
            pca = PCA(n_components=min(50, dim), svd_solver='randomized')
            pca.fit(dataset)
            # Use 95% variance threshold for early stopping
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cum_var >= 0.95) + 1
            if n_components < min(50, dim):
                dataset = pca.transform(dataset)[:, :n_components]
            else:
                dataset = pca.transform(dataset)
        # Configure HDBSCAN with optimized parameters
        algorithm = 'prims_kdtree' if n < 10000 else 'boruvka_kdtree'
        leaf_size = 30
        
        # Perform optimized HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            algorithm=algorithm,
            approx_min_span_tree=True,
            gen_min_span_tree=False,
            core_dist_n_jobs=-1,
            leaf_size=leaf_size,
            prediction_data=False
        )
        clusterer.fit(dataset)
        
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_
        
        num_clusters = len(set(labels[labels != -1]))
        num_noise_points = int(np.sum(labels == -1))
        
        return {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points
        }