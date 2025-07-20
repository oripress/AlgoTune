import numpy as np
from typing import Any
import faiss

class Solver:
    def __init__(self):
        # Pre-compile/warm up
        pass
    
    def solve(self, problem: dict[str, Any]) -> list[int]:
        """Ultra-fast K-means clustering using FAISS."""
        X = np.ascontiguousarray(problem["X"], dtype=np.float32)
        k = problem["k"]
        n_samples = len(X)
        
        # Handle edge cases
        if k >= n_samples:
            return list(range(n_samples))
        if k == 1:
            return [0] * n_samples
        
        # Use FAISS K-means with minimal iterations
        # spherical=True uses cosine distance which can be faster
        # nredo=1 to avoid multiple runs
        # max_points_per_centroid to handle imbalanced clusters
        n_features = X.shape[1]
        
        kmeans = faiss.Kmeans(
            n_features, 
            k, 
            niter=5,  # Very few iterations
            nredo=1,  # Single run
            verbose=False,
            gpu=False,
            seed=1234,
            spherical=False,
            max_points_per_centroid=n_samples,  # Allow imbalanced clusters
            min_points_per_centroid=0
        )
        
        # Train the model
        kmeans.train(X)
        
        # Get cluster assignments
        _, labels = kmeans.index.search(X, 1)
        
        return labels.ravel().tolist()