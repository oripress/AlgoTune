import numpy as np
import faiss
import hdbscan
class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Extract and prepare dataset
        data_list = problem.get("dataset", [])
        if not data_list:
            return {
                "labels": [],
                "probabilities": [],
                "cluster_persistence": [],
                "num_clusters": 0,
                "num_noise_points": 0,
            }
        dataset = np.asarray(data_list, dtype=np.float64)
        if dataset.ndim == 1:
            dataset = dataset.reshape(-1, 1)
        n, dim = dataset.shape
        # Clustering parameters
        min_cluster_size = problem.get("min_cluster_size", 5)
        min_samples = problem.get("min_samples", 3)
        # Threshold for sampling to speed up large datasets
        threshold = 2000
        if n <= threshold:
            # Full HDBSCAN clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                core_dist_n_jobs=-1,
                approx_min_span_tree=True,
            )
            clusterer.fit(dataset)
            labels = clusterer.labels_
            probabilities = clusterer.probabilities_
            persistence = clusterer.cluster_persistence_.tolist()
        else:
            # Sample subset for clustering
            rng = np.random.default_rng(0)
            sample_idx = rng.choice(n, threshold, replace=False)
            sample_data = dataset[sample_idx]
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                core_dist_n_jobs=-1,
                approx_min_span_tree=True,
            )
            clusterer.fit(sample_data)
            sample_labels = clusterer.labels_
            sample_prob = clusterer.probabilities_
            persistence = clusterer.cluster_persistence_.tolist()
            # Assign clusters to all points via nearest neighbor
            # Assign clusters to all points via nearest neighbor using Faiss
            sample32 = sample_data.astype(np.float32, order='C')
            data32 = dataset.astype(np.float32, order='C')
            index = faiss.IndexFlatL2(dim)
            index.add(sample32)
            _, I = index.search(data32, 1)
            nn = I[:, 0]
            labels = sample_labels[nn]
            probabilities = sample_prob[nn]
        labels = np.asarray(labels, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        num_clusters = int(len(set(labels[labels != -1])))
        num_noise_points = int(np.sum(labels == -1))
        # Build solution
        return {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence,
            "num_clusters": num_clusters,
            "num_noise_points": num_noise_points,
        }