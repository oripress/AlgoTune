import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve the Vector Quantization problem using optimized Faiss operations.
        """
        vectors = np.array(problem["vectors"], dtype=np.float32)
        k = problem["k"]
        dim = vectors.shape[1]

        kmeans = faiss.Kmeans(dim, k, niter=58, nredo=1, verbose=False)
        kmeans.train(vectors)
        
        # Reuse the index from kmeans instead of creating a new one
        distances, assignments = kmeans.index.search(vectors, 1)

        quantization_error = float(np.mean(distances))
        return {
            "centroids": kmeans.centroids.tolist(),
            "assignments": assignments[:, 0].tolist(),
            "quantization_error": quantization_error,
        }