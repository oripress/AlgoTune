import numpy as np
import hdbscan
from typing import Any
import warnings
warnings.filterwarnings('ignore')

class Solver:
    def __init__(self):
        """Initialize solver with optimized settings."""
        pass
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the clustering problem using optimized HDBSCAN.
        """
        # Extract problem parameters
        dataset = np.array(problem["dataset"], dtype=np.float32)  # Use float32 for speed
        n_points = len(dataset)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        
        # Choose algorithm based on dataset size
        if n_points < 1000:
            algorithm = 'best'
        elif n_points < 5000:
            algorithm = 'prims_kdtree'
        else:
            algorithm = 'boruvka_kdtree'  # Faster for large datasets
        
        # Optimize leaf_size based on dataset size
        leaf_size = min(40, max(20, n_points // 100))
        
        # Perform HDBSCAN clustering with optimized parameters
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            algorithm=algorithm,
            leaf_size=leaf_size,
            core_dist_n_jobs=-1,  # Use all available cores
            gen_min_span_tree=False,  # Don't generate MST if not needed
            prediction_data=False,  # Don't generate prediction data
            approx_min_span_tree=True,  # Use approximation for large datasets
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