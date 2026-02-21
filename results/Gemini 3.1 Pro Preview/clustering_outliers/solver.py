import numpy as np
from typing import Any
from sklearn.cluster import HDBSCAN

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        dataset = np.array(problem["dataset"], dtype=np.float64)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=min_samples,
            algorithm='auto',
            n_jobs=8
        )
        clusterer.fit(dataset)
        
        labels = clusterer.labels_
        
        return {
            "labels": labels.tolist(),
            "probabilities": clusterer.probabilities_.tolist(),
            "cluster_persistence": clusterer.cluster_persistence_.tolist() if hasattr(clusterer, 'cluster_persistence_') else [],
            "num_clusters": int(labels.max()) + 1 if len(labels) > 0 else 0,
            "num_noise_points": int(np.sum(labels == -1)),
        }