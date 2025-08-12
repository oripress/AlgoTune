import numpy as np
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solve the Vector Quantization problem using the Faiss library with optimized error computation.

        This method implements vector quantization using Faiss's kmeans clustering with
        optimized quantization error computation for better performance.

        :param problem: A dictionary representing the Vector Quantization problem.
        :return: A dictionary with keys:
                 "centroids": 2D list representing the k centroids/codewords found.
                 "assignments": 1D list indicating which centroid each input vector is assigned to.
                 "quantization_error": The mean squared error of the quantization.
        """
        vectors = problem["vectors"]
        k = problem["k"]
        # Convert to numpy array with optimal settings
        vectors = np.array(vectors, dtype=np.float32)
        dim = vectors.shape[1]
        
        # Use Faiss Kmeans with optimized parameters
        kmeans = faiss.Kmeans(dim, k, niter=60, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids
        
        # Use the already trained index for assignments
        distances, assignments = kmeans.index.search(vectors, 1)
        
        # Optimized quantization error computation
        quantization_error = np.sum(distances) / len(vectors)
        
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": float(quantization_error),
        }