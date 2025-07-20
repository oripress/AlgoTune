import numpy as np
import hdbscan
from typing import Any
from sklearn.cluster import DBSCAN
import numba as nb

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the clustering problem using optimized HDBSCAN.
        
        :param problem: A dictionary representing the clustering problem.
        :return: A dictionary with clustering solution details
        """
        # Extract problem parameters
        dataset = np.asarray(problem["dataset"], dtype=np.float64, order='C')
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        
        # Use the most optimized HDBSCAN settings
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            algorithm='prims_kdtree',  # Try prims algorithm
            core_dist_n_jobs=-1,  # Use all cores
            approx_min_span_tree=True,  # Use approximation for speed
            leaf_size=20,  # Smaller leaf size for faster queries
            metric='euclidean',
            gen_min_span_tree=False,
            prediction_data=False
        )
        
        clusterer.fit(dataset)
        
        # Directly access internal arrays to avoid copies
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_
        
        # Optimize the conversion to lists and counting
        labels_list = labels.tolist()
        num_noise = int((labels == -1).sum())
        unique_clusters = np.unique(labels[labels != -1])
        num_clusters = len(unique_clusters)
        
        # Prepare solution
        solution = {
            "labels": labels_list,
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": num_clusters,
            "num_noise_points": num_noise
        }
        
        return solution