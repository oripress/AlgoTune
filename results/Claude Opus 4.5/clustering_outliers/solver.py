import numpy as np
from sklearn.cluster import HDBSCAN

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the clustering problem using sklearn's HDBSCAN.
        """
        dataset = np.array(problem["dataset"], dtype=np.float64)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        
        n, dim = dataset.shape
        
        # Choose algorithm based on data characteristics
        if dim <= 15:
            algorithm = 'kd_tree'
        elif dim <= 30:
            algorithm = 'ball_tree'
        else:
            algorithm = 'brute'
        
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=min_samples,
            algorithm=algorithm,
            n_jobs=-1,
            store_centers=None
        )
        clusterer.fit(dataset)
        
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        
        # Extract cluster persistence from the condensed tree if available
        persistence = []
        if hasattr(clusterer, 'cluster_persistence_'):
            persistence = clusterer.cluster_persistence_.tolist()
        
        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence,
            "num_clusters": len(set(labels[labels != -1])),
            "num_noise_points": int(np.sum(labels == -1)),
        }
        return solution