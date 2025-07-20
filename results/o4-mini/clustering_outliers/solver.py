import numpy as np
from sklearn.cluster import DBSCAN
import faiss

class Solver:
    def solve(self, problem: dict, **kwargs) -> dict:
        # Load dataset
        data_list = problem.get("dataset", [])
        data = np.asarray(data_list, dtype=np.float32)
        n = data.shape[0]
        # Empty case
        if n == 0:
            return {
                "labels": [],
                "probabilities": [],
                "cluster_persistence": [],
                "num_clusters": 0,
                "num_noise_points": 0
            }
        # DBSCAN parameter
        min_samples = int(problem.get("min_samples", 3))
        # Sampling threshold
        SAMPLE = 2000

        # Estimate squared eps using faiss k-NN on X
        def estimate_eps2(X: np.ndarray) -> float:
            m, dim = X.shape
            index = faiss.IndexFlatL2(dim)
            index.add(X)
            # k = min_samples+1 for self-distance at zero
            k = min(min_samples + 1, m)
            D, _ = index.search(X, k)
            # squared distance to k-th neighbor
            arr = D[:, k - 1]
            return float(np.median(arr))

        # Choose strategy
        if n <= SAMPLE:
            # Full DBSCAN on all points
            eps2 = estimate_eps2(data)
            eps = np.sqrt(eps2)
            db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(data)
            labels = db.labels_
            probabilities = np.where(labels == -1, 0.0, 1.0)
            persistence = []
        else:
            # Sample subset for DBSCAN
            rng = np.random.default_rng(42)
            idx = rng.choice(n, SAMPLE, replace=False)
            sample = data[idx]
            eps2 = estimate_eps2(sample)
            eps = np.sqrt(eps2)
            # Cluster the sample
            db_s = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(sample)
            labels_s = db_s.labels_
            # Build faiss index on sample
            dim = sample.shape[1]
            index = faiss.IndexFlatL2(dim)
            index.add(sample)
            # Assign each point to nearest sample neighbor
            D, I = index.search(data, 1)
            D0 = D.ravel()
            I0 = I.ravel()
            mask = (D0 <= eps2) & (labels_s[I0] != -1)
            labels = np.full(n, -1, dtype=int)
            labels[mask] = labels_s[I0[mask]]
            probabilities = np.where(labels == -1, 0.0, 1.0)
            persistence = []

        # Compute stats
        num_noise = int((labels == -1).sum())
        unique = set(int(x) for x in labels.tolist() if x != -1)
        num_clusters = len(unique)

        # Return solution
        return {
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
            "cluster_persistence": persistence,
            "num_clusters": num_clusters,
            "num_noise_points": num_noise
        }