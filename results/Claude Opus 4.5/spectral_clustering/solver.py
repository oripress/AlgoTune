import numpy as np
from sklearn.cluster import SpectralClustering

class Solver:
    def solve(self, problem, **kwargs):
        similarity_matrix = problem["similarity_matrix"]
        n_clusters = problem["n_clusters"]
        
        if not isinstance(similarity_matrix, np.ndarray):
            similarity_matrix = np.array(similarity_matrix)
        
        if (similarity_matrix.ndim != 2 or 
            similarity_matrix.shape[0] != similarity_matrix.shape[1]):
            raise ValueError("Invalid similarity matrix provided.")
        if not isinstance(n_clusters, int) or n_clusters < 1:
            raise ValueError("Invalid number of clusters provided.")
        
        n = similarity_matrix.shape[0]
        
        if n_clusters >= n:
            labels = np.arange(n)
        elif n == 0:
            labels = np.array([], dtype=int)
        else:
            try:
                # Use cluster_qr which is faster and avoids some edge case issues
                model = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    assign_labels="cluster_qr",
                    random_state=42,
                )
                labels = model.fit_predict(similarity_matrix)
            except Exception:
                labels = np.zeros(n, dtype=int)
        
        return {"labels": labels}