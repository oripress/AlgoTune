import numpy as np
from scipy.linalg import eigh
import faiss
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fast spectral clustering using FAISS for k-means."""
        similarity_matrix = np.asarray(problem["similarity_matrix"], dtype=np.float32)
        n_clusters = problem["n_clusters"]
        n_samples = len(similarity_matrix)
        
        # Handle edge cases
        if n_samples == 0:
            return {"labels": [], "n_clusters": n_clusters}
        
        if n_clusters >= n_samples:
            return {"labels": list(range(n_samples)), "n_clusters": n_clusters}
        
        if n_clusters == 1:
            return {"labels": [0] * n_samples, "n_clusters": n_clusters}
        
        # Fast computation of normalized Laplacian using vectorized operations
        degrees = np.sum(similarity_matrix, axis=1)
        degrees = np.maximum(degrees, 1e-10)
        d_sqrt_inv = 1.0 / np.sqrt(degrees)
        
        # Vectorized normalized Laplacian computation
        normalized_laplacian = np.eye(n_samples, dtype=np.float32)
        normalized_laplacian -= (d_sqrt_inv[:, None] * similarity_matrix * d_sqrt_inv[None, :])
        
        # Compute eigenvectors efficiently
        # Only compute the needed eigenvectors
        eigenvalues, eigenvectors = eigh(normalized_laplacian, 
                                        subset_by_index=[0, n_clusters-1])
        
        # Normalize rows efficiently
        eigenvectors = eigenvectors.astype(np.float32)
        row_norms = np.linalg.norm(eigenvectors, axis=1, keepdims=True)
        row_norms = np.maximum(row_norms, 1e-10)
        eigenvectors /= row_norms
        
        # Use FAISS for ultra-fast k-means
        d = eigenvectors.shape[1]
        kmeans = faiss.Kmeans(d, n_clusters, 
                              niter=20,
                              nredo=3,
                              seed=42,
                              spherical=False,
                              gpu=False)
        
        # Train k-means
        kmeans.train(eigenvectors)
        
        # Get cluster assignments
        _, labels = kmeans.index.search(eigenvectors, 1)
        labels = labels.flatten().astype(np.int32)
        
        return {"labels": labels.tolist(), "n_clusters": n_clusters}