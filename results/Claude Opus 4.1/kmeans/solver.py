import numpy as np
from typing import Any
import faiss

class Solver:
    def __init__(self):
        # Pre-compile/initialize anything needed
        pass
    
    def solve(self, problem: dict[str, Any], **kwargs) -> list[int]:
        """K-means clustering implementation using FAISS."""
        X = np.array(problem["X"], dtype=np.float32)
        k = problem["k"]
        n = len(X)
        
        # Fallback for edge cases
        if k <= 0 or n == 0:
            return [0] * n
        if k >= n:
            return list(range(n))
        
        # Use FAISS for fast K-means with optimized parameters
        d = X.shape[1]  # dimension
        
        # Create and train kmeans with minimal iterations for speed
        kmeans = faiss.Kmeans(d, k, niter=10, verbose=False, nredo=1, 
                              spherical=False, int_centroids=False, 
                              update_index=False, seed=1234)
        kmeans.train(X)
        
        # Assign labels
        _, labels = kmeans.index.search(X, 1)
        return labels.flatten().tolist()