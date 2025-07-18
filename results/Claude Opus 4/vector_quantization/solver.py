import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        """Solve the Vector Quantization problem using Faiss library."""
        vectors = np.array(problem["vectors"], dtype=np.float32)
        k = problem["k"]
        dim = vectors.shape[1]
        
        # Use Faiss kmeans clustering
        kmeans = faiss.Kmeans(dim, k, niter=100, verbose=False)
        kmeans.train(vectors)
        centroids = kmeans.centroids
        
        # Find nearest centroid for each vector
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