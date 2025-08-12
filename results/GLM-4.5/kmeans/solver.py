from typing import Any
import numpy as np
from sklearn.cluster import KMeans

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        """Ultra-fast K-means clustering using scikit-learn with maximum optimization."""
        X = np.array(problem["X"])
        k = problem["k"]
        n_samples = X.shape[0]
        
        # Handle edge cases
        if n_samples == 0:
            return []
        if k <= 0:
            return [0] * n_samples
        if k >= n_samples:
            return list(range(n_samples))
        
        # Use scikit-learn's KMeans with optimal balance of speed and accuracy
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=3, init='random', 
                       algorithm='lloyd', random_state=42, tol=0.1)
        kmeans.fit(X)
        
        return kmeans.labels_.tolist()