import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        """
        Fast vector quantization: reduced-iteration FAISS KMeans.
        """
        # Load data
        vectors = np.array(problem["vectors"], dtype=np.float32)
        n, dim = vectors.shape
        k = int(problem["k"])

        # Run KMeans on full data with fewer iterations
        kmeans = faiss.Kmeans(dim, k, niter=58, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids  # shape (k, dim)

        # Assign each vector to the nearest centroid
        index = faiss.IndexFlatL2(dim)
        index.add(centroids)
        distances, assignments = index.search(vectors, 1)
        assignments = assignments.flatten().astype(int)
        distances = distances.flatten()

        # Compute mean squared error
        quant_error = float(np.mean(distances))
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.tolist(),
            "quantization_error": quant_error
        }