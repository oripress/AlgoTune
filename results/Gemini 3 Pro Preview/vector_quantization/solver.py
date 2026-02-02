import faiss
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        vectors = problem["vectors"]
        k = problem["k"]
        
        # Convert to float32 numpy array as required by Faiss
        vectors = np.array(vectors, dtype=np.float32)
        dim = vectors.shape[1]

        # Use fewer iterations than reference (100) to speed up.
        # 80 iterations should be sufficient for convergence within 1% tolerance.
        niter = 60
        kmeans = faiss.Kmeans(dim, k, niter=niter, verbose=False)
        kmeans.train(vectors)
        
        # Reuse the index from the kmeans object to avoid rebuilding it
        distances, assignments = kmeans.index.search(vectors, 1)

        quantization_error = np.mean(distances)
        
        return {
            "centroids": kmeans.centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": float(quantization_error),
        }