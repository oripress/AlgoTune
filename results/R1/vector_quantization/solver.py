import numpy as np
import faiss

class Solver:
    def solve(self, problem, **kwargs):
        vectors = problem["vectors"]
        k = problem["k"]
        vectors = np.array(vectors).astype(np.float32)
        dim = vectors.shape[1]
        n_vectors = vectors.shape[0]
        
        # Handle case where we don't have enough vectors
        if n_vectors < k:
            centroids = vectors.tolist()
            last_centroid = vectors[-1].tolist() if n_vectors > 0 else [0.0]*dim
            centroids.extend([last_centroid] * (k - n_vectors))
            assignments = list(range(n_vectors)) if n_vectors > 0 else []
            quantization_error = 0.0
            return {
                "centroids": centroids,
                "assignments": assignments,
                "quantization_error": quantization_error,
            }
        else:
            # Use the exact same Faiss Kmeans implementation as reference
            # but with reduced number of iterations for better performance
            kmeans = faiss.Kmeans(dim, k, niter=58, verbose=False)
            kmeans.train(vectors)
            centroids = kmeans.centroids
            
            # Use the same assignment method as reference to ensure quality
            index = faiss.IndexFlatL2(dim)
            index.add(centroids)
            distances, assignments = index.search(vectors, 1)
            
            # Calculate quantization error the same way as reference
            quantization_error = np.mean(distances)
        
        return {
            "centroids": centroids.tolist(),
            "assignments": assignments.flatten().tolist(),
            "quantization_error": float(quantization_error),
        }