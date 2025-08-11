from typing import Any, Dict

import numpy as np
import hdbscan

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Perform HDBSCAN clustering on the given dataset.

        Returns:
            dict with keys:
                - "labels": list[int]
                - "probabilities": list[float]
                - "cluster_persistence": list[float]
        """
        dataset = problem.get("dataset", [])
        if not dataset:
            # Empty dataset: return empty outputs with required keys
            return {
                "labels": [],
                "probabilities": [],
                "cluster_persistence": [],
            }

        # Retrieve parameters
        min_cluster_size = int(problem.get("min_cluster_size", 5))
        min_samples = int(problem.get("min_samples", 3))

        n = len(dataset)

        # Fast path: if fewer points than min_cluster_size, HDBSCAN yields all noise
        if n < min_cluster_size:
            labels = [-1] * n
            probabilities = [0.0] * n
            return {
                "labels": labels,
                "probabilities": probabilities,
                "cluster_persistence": [],
            }

        # Fast path: all points identical -> single dense cluster
        first = dataset[0]
        all_identical = True
        for i in range(1, n):
            if dataset[i] != first:
                all_identical = False
                break
        if all_identical:
            labels = [0] * n
            probabilities = [1.0] * n
            return {
                "labels": labels,
                "probabilities": probabilities,
                "cluster_persistence": [1.0],
            }

        # Use provided dimension if available to select algorithm without first materializing array
        d = int(problem.get("dim", 0)) or None

        # Convert to numpy array; ensure 2D shape and contiguous memory for speed (float32 to reduce work)
        X = np.asarray(dataset, dtype=np.float32, order="C")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if d is None:
            d = X.shape[1]

        # Choose algorithm (KD-tree boruvka is fast for low/moderate dims)
        algo = "boruvka_kdtree" if d <= 15 else "best"
        # Parallelism: avoid overhead for small n, enable for larger datasets
        core_jobs = -1 if n >= 4096 else 1

        # HDBSCAN clustering with faster settings where possible
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            algorithm=algo,
            approx_min_span_tree=True,   # speed up (default)
            gen_min_span_tree=False,     # avoid extra work
            core_dist_n_jobs=core_jobs,  # adapt parallelism to problem size
            metric="euclidean",
        )
        clusterer.fit(X)

        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        return {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
        }