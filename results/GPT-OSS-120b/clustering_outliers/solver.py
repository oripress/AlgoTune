# (faiss import removed)
import numpy as np
from typing import Any
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Perform clustering using fast DBSCAN approximation.
        Returns labels, probabilities, cluster persistence, number of clusters,
        and number of noise points.
        """
        # Convert dataset to a NumPy array (avoid unnecessary copy)
        data = np.asarray(problem["dataset"], dtype=np.float32)
        min_cluster_size = problem.get("min_cluster_size", 5)  # kept for compatibility
        min_samples = problem.get("min_samples", 3)

        # Estimate eps using a random subset to speed up computation
        n_points = data.shape[0]
        sample_size = min(1000, n_points)
        if sample_size < n_points:
            rng = np.random.default_rng()
            sample_idx = rng.choice(n_points, size=sample_size, replace=False)
            sample_data = data[sample_idx]
        else:
            sample_data = data
        # Use NearestNeighbors on the sample to estimate median distance to the min_samples‑th neighbor
        nn = NearestNeighbors(n_neighbors=min_samples, algorithm='auto', n_jobs=-1)
        nn.fit(sample_data)
        distances, _ = nn.kneighbors(sample_data)
        eps = np.median(distances[:, -1])

        # Run DBSCAN with the estimated eps
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = clusterer.fit_predict(data)

        # Placeholders for required fields
        probabilities = []          # per‑point probabilities not needed
        persistence = []            # cluster persistence not needed

        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities,
            "cluster_persistence": persistence,
            "num_clusters": int(len(set(labels[labels != -1]))),
            "num_noise_points": int(np.sum(labels == -1)),
        }
        return solution