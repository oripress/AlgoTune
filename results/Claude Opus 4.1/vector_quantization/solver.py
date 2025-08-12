import numpy as np
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Vector Quantization problem using optimized Faiss k-means.
        """
        vectors = np.array(problem["vectors"], dtype=np.float32)
        k = problem["k"]
        dim = vectors.shape[1]
        
        # Use Faiss for optimized k-means clustering
        # Reduce iterations for speed while maintaining quality
        kmeans = faiss.Kmeans(
            dim, 
            k, 
            niter=20,  # Reduced iterations for speed
            nredo=1,   # Single run (no multiple attempts)
            verbose=False,
            spherical=False,
            min_points_per_centroid=1,
            max_points_per_centroid=vectors.shape[0]
        )
        
        # Train k-means
        kmeans.train(vectors)
        centroids = kmeans.centroids
        
        # Use efficient index for assignment
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)
        
        # Calculate quantization error
        quantization_error = np.mean(distances)
        
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": float(quantization_error)
        }