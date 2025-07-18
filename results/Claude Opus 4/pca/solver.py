import numpy as np
from typing import Any
import numba as nb

@nb.njit(fastmath=True, cache=True)
def pca_svd_small_n(X_centered, n_components):
    """When n < m, use covariance matrix approach."""
    # Compute covariance matrix X^T X
    cov = X_centered.T @ X_centered
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort by eigenvalues in descending order
    idx = eigvals.argsort()[::-1]
    # Return top n_components eigenvectors (transposed)
    return eigvecs[:, idx[:n_components]].T

@nb.njit(fastmath=True, cache=True)
def pca_svd_small_m(X_centered, n_components):
    """When m < n, use Gram matrix approach."""
    # Compute Gram matrix X X^T
    gram = X_centered @ X_centered.T
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(gram)
    # Sort by eigenvalues in descending order
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Convert eigenvectors of XX^T to eigenvectors of X^TX
    V = X_centered.T @ eigvecs[:, :n_components]
    # Normalize
    for i in range(n_components):
        V[:, i] /= np.linalg.norm(V[:, i])
    return V.T

class Solver:
    def solve(self, problem: dict[str, Any]) -> np.ndarray:
        """Fast PCA implementation using optimized SVD approaches."""
        X = np.asarray(problem["X"], dtype=np.float64)
        n_components = problem["n_components"]
        
        # Center the data
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        
        m, n = X_centered.shape
        
        # Choose the most efficient approach
        if n_components >= min(m, n):
            # Need full SVD
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            return Vt[:n_components]
        elif n <= m:
            # Use covariance matrix approach
            return pca_svd_small_n(X_centered, n_components)
        else:
            # Use Gram matrix approach
            return pca_svd_small_m(X_centered, n_components)