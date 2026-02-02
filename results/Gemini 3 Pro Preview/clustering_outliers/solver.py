import hdbscan
import numpy as np
import psutil

class Solver:
    def solve(self, problem, **kwargs):
        # Use float32 for speed
        dataset = np.array(problem["dataset"], dtype=np.float32)
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)

        n = len(dataset)
        # Only use parallel processing for larger datasets to avoid overhead
        if n > 2000:
            n_jobs = psutil.cpu_count(logical=False) or 1
        else:
            n_jobs = 1

        # Enable approximate MST for speedup
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, 
            min_samples=min_samples,
            core_dist_n_jobs=n_jobs,
            approx_min_span_tree=True,
            algorithm='prims_kdtree',
            leaf_size=30,
            cluster_selection_method='leaf'
        )
        clusterer.fit(dataset)

        labels = clusterer.labels_
        probabilities = clusterer.probabilities_
        persistence = clusterer.cluster_persistence_

        solution = {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence.tolist(),
            "num_clusters": len(set(labels[labels != -1])),
            "num_noise_points": int(np.sum(labels == -1)),
        }
        return solution