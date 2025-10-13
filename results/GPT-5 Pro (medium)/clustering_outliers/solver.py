from typing import Any

import numpy as np
import hdbscan


class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> Any:
        """
        Clustering with HDBSCAN, robust to outliers.

        Parameters
        ----------
        problem : dict
            {
                "n": int,
                "dim": int,
                "dataset": list[list[float]] | np.ndarray,
                "min_cluster_size": int (optional, default=5),
                "min_samples": int (optional, default=3),
            }

        Returns
        -------
        dict
            {
                "labels": list[int],
                "probabilities": list[float],
                "cluster_persistence": list[float],
                "num_clusters": int,
                "num_noise_points": int,
            }
        """
        # Extract problem parameters
        dataset = np.asarray(problem["dataset"], dtype=float)
        min_cluster_size = int(problem.get("min_cluster_size", 5))
        min_samples_value = problem.get("min_samples", 3)
        min_samples = None if min_samples_value is None else int(min_samples_value)

        # Handle empty dataset gracefully
        n_points = dataset.shape[0]
        if n_points == 0:
            return {
                "labels": [],
                "probabilities": [],
                "cluster_persistence": [],
                "num_clusters": 0,
                "num_noise_points": 0,
            }

        # Perform HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        clusterer.fit(dataset)

        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        # Prepare solution including required fields for validation
        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": int(len(set(labels[labels != -1]))),
            "num_noise_points": int(np.sum(labels == -1)),
        }
        return solution