from typing import Any
import numpy as np
import numba
from numba import prange

@numba.jit(nopython=True, fastmath=True, cache=True)
def kmeans_numba(X, k, max_iter=100, tol=1e-4, seed=42):
    np.random.seed(seed)
    n, d = X.shape
    
    # Random initialization
    # We pick k random indices
    indices = np.random.choice(n, k, replace=False)
    centroids = X[indices].copy()
    
    labels = np.zeros(n, dtype=np.int32)
    distances = np.zeros(n, dtype=np.float32)
    
    for _ in range(max_iter):
        # E-step: assign points to nearest centroid
        changed = False
        for i in range(n):
            min_dist = np.inf
            best_k = -1
            for j in range(k):
                dist = 0.0
                for l in range(d):
                    diff = X[i, l] - centroids[j, l]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
                    best_k = j
            
            if labels[i] != best_k:
                changed = True
                labels[i] = best_k
        
        if not changed:
            break
            
        # M-step: update centroids
        new_centroids = np.zeros((k, d), dtype=np.float32)
        counts = np.zeros(k, dtype=np.int32)
        
        for i in range(n):
            c = labels[i]
            counts[c] += 1
            for l in range(d):
                new_centroids[c, l] += X[i, l]
        
        for j in range(k):
            if counts[j] > 0:
                for l in range(d):
                    new_centroids[j, l] /= counts[j]
            else:
                # Re-initialize empty cluster to a random point
                # This is a simple heuristic
                rand_idx = np.random.randint(0, n)
                for l in range(d):
                    new_centroids[j, l] = X[rand_idx, l]

        # Check convergence based on centroid movement could be added here
        # But checking labels change is often enough for discrete convergence
        
        centroids = new_centroids

    return labels

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[int]:
        X = np.array(problem["X"], dtype=np.float32)
        k = problem["k"]
        # Ensure k is not larger than n
        if k > len(X):
            k = len(X)
            
        labels = kmeans_numba(X, k)
        return list(labels)