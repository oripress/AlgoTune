import numpy as np
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        vectors = np.array(problem["vectors"], dtype=np.float32)
        k = problem["k"]
        n_vectors = vectors.shape[0]
        dim = vectors.shape[1]
        
        # Use spherical=False, fewer iters with nredo for quality
        # max_points_per_centroid limits training set size per iteration
        kmeans = faiss.Kmeans(dim, k, niter=50, nredo=2, verbose=False,
                             max_points_per_centroid=256)
        kmeans.train(vectors)
        centroids = kmeans.centroids
        
        # Use kmeans index directly
        distances, assignments = kmeans.index.search(vectors, 1)
        
        quantization_error = float(np.mean(distances))
        
        return {
            "centroids": centroids,
            "assignments": assignments.ravel(),
            "quantization_error": quantization_error,
        }