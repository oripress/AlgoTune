import numpy as np
from typing import Any
import hdbscan

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the clustering problem using HDBSCAN with optimized settings.
        """
        # Extract problem parameters
        dataset = np.array(problem["dataset"], dtype=np.float32)  # Use float32 for speed
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        
        # Perform HDBSCAN clustering with optimized parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            algorithm='best',  # Let HDBSCAN choose the best algorithm
            leaf_size=30,
            core_dist_n_jobs=1  # Single thread can be faster for small datasets
        )
        clusterer.fit(dataset)
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_
        
        # Prepare solution
        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": len(set(labels[labels != -1])),
            "num_noise_points": int(np.sum(labels == -1)),
        }
        return solution