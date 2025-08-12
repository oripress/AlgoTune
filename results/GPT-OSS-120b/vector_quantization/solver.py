import numpy as np
import faiss
from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform vector quantization using Faiss's K‑means clustering.

        Parameters
        ----------
        problem : dict
            Dictionary containing:
                - "vectors": list of vectors (list of list of floats)
                - "k": number of centroids
                - other keys are ignored.

        Returns
        -------
        dict
            {
                "centroids": list of k centroids (list of list of floats),
                "assignments": list of length n_vectors with cluster indices,
                "quantization_error": mean squared Euclidean distance.
            }
        """
        # Extract and convert vectors
        vectors = np.array(problem["vectors"], dtype=np.float32)
        k = int(problem["k"])
        n, dim = vectors.shape

        # Faiss K‑means
        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids  # shape (k, dim)

        # Assign each vector to nearest centroid
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)  # distances shape (n,1)

        # Compute mean squared error (mean of squared distances)
        quantization_error = float(np.mean(distances))

        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.squeeze().tolist(),
            "quantization_error": quantization_error,
        }