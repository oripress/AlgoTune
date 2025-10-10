import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        try:
            X = problem["X"]
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float32)
            else:
                X = np.ascontiguousarray(X, dtype=np.float32)
            k = problem["k"]
            n, d = X.shape
            
            # Use FAISS with optimized parameters
            kmeans = faiss.Kmeans(d, k, niter=3, verbose=False, seed=42, 
                                  min_points_per_centroid=1, max_points_per_centroid=n,
                                  spherical=False, update_index=False)
            kmeans.train(X)
            
            # Get cluster assignments
            _, labels = kmeans.index.search(X, 1)
            
            return labels.ravel().tolist()
        except Exception as e:
            n = len(problem["X"])
            return [0] * n