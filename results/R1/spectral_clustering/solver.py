import numpy as np
import faiss
import scipy.linalg
import os
from scipy.sparse.linalg import eigsh

# Configure environment for optimal performance
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

def normalize_affinity(A, D_inv_sqrt):
    """BLAS-accelerated affinity matrix normalization with in-place operations."""
    # Efficient broadcasting for normalization
    A = A * D_inv_sqrt[:, None]
    A = A * D_inv_sqrt[None, :]
    return A

def normalize_rows(U):
    """Efficient row normalization with safe division and in-place operations."""
    norms = np.sqrt(np.sum(U**2, axis=1, keepdims=True))
    np.maximum(norms, 1e-12, out=norms)
    np.divide(U, norms, out=U)
    return U

class Solver:
    def solve(self, problem, **kwargs):
        # Convert input to float32 for faster computation
        A = np.array(problem["similarity_matrix"], dtype=np.float32)
        n_samples = A.shape[0]
        n_clusters = problem["n_clusters"]
        
        # Handle edge cases efficiently
        if n_clusters >= n_samples:
            return {"labels": list(range(n_samples))}
        if n_samples == 0:
            return {"labels": []}
        if n_clusters == 1:
            return {"labels": [0] * n_samples}
        
        # Compute degree vector using optimized vectorized sum
        D = np.sum(A, axis=1)
        
        # Safe division with vectorized operations
        D_safe = np.where(D > 0, D, 1.0)
        D_inv_sqrt = 1.0 / np.sqrt(D_safe).astype(np.float32)
        
        # Normalize affinity matrix in-place
        A = normalize_affinity(A, D_inv_sqrt)
        
        # Form the normalized Laplacian: L = I - A (because A now is D^{-1/2} A D^{-1/2})
        # But note: we want the eigenvectors of A (which is the normalized affinity) for the largest eigenvalues.
        # Actually, the spectral clustering uses the eigenvectors of the normalized affinity for the top eigenvalues.
        # So we can avoid forming the Laplacian and compute the top eigenvectors of A.
        # We use eigsh for the symmetric matrix A to get the top eigenvalues and eigenvectors.
        if n_clusters < n_samples:
            try:
                # Compute the largest eigenvalues of A (which is the normalized affinity)
                eigenvalues, eigenvectors = eigsh(
                    A, 
                    k=n_clusters, 
                    which='LM', 
                    maxiter=2000,
                    tol=1e-4  # Increased tolerance for faster convergence
                )
                # Sort eigenvalues in descending order and eigenvectors accordingly
                idx = np.argsort(eigenvalues)[::-1]
                U = eigenvectors[:, idx].astype(np.float32)
            except:
                # Fallback to dense eigensolver
                eigenvalues, eigenvectors = scipy.linalg.eigh(A)
                idx = np.argsort(eigenvalues)[::-1][:n_clusters]
                U = eigenvectors[:, idx].astype(np.float32)
        else:
            U = np.eye(n_samples, dtype=np.float32)
        
        # Normalize rows in-place
        U = normalize_rows(U)
        
        # Configure Faiss for maximum performance
        faiss.omp_set_num_threads(8)  # Use 8 threads for k-means
        niter = 2  # Reduced iterations
        nredo = 1
        max_points = min(32768, n_samples // max(1, n_clusters))
        
        # Use Faiss with optimized parameters
        kmeans = faiss.Kmeans(
            d=U.shape[1], 
            k=n_clusters, 
            niter=niter, 
            nredo=nredo, 
            seed=42,
            spherical=True,
            verbose=False,
            max_points_per_centroid=max_points,
            gpu=False
        )
        kmeans.train(U)
        _, labels = kmeans.index.search(U, 1)
        labels = labels.ravel().astype(int)
        
        return {"labels": labels.tolist()}