import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        vectors = problem["vectors"]
        k = problem["k"]
        vectors = np.array(vectors, dtype=np.float32)
        n_vectors, dim = vectors.shape

        # 60 iterations balances speed and quality
        kmeans = faiss.Kmeans(dim, k, niter=60, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids

        distances, assignments = kmeans.index.search(vectors, 1)

        quantization_error = float(np.mean(distances))
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.ravel().tolist(),
            "quantization_error": quantization_error,
        }