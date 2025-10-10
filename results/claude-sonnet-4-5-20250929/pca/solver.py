import numpy as np
from scipy.linalg import eigh, svd

class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=np.float64)
        n_components = problem["n_components"]
        m, n = X.shape
        
        # Center the data
        X_centered = X - np.mean(X, axis=0)
        
        # Choose method based on data dimensions
        if n_components < min(m, n) * 0.8:
            # Use covariance matrix approach for small n_components
            if m >= n:
                # Compute X^T X / (m-1)
                cov = X_centered.T @ X_centered
                # Get top eigenvalues and eigenvectors
                eigenvalues, eigenvectors = eigh(cov)
                # Sort in descending order
                idx = np.argsort(eigenvalues)[::-1]
                # Return top n_components eigenvectors as rows
                return eigenvectors[:, idx[:n_components]].T
            else:
                # For m < n, use X X^T and compute V from U
                gram = X_centered @ X_centered.T
                eigenvalues, U = eigh(gram)
                idx = np.argsort(eigenvalues)[::-1]
                U = U[:, idx[:n_components]]
                # V = X^T U / S
                S = np.sqrt(eigenvalues[idx[:n_components]])
                Vt = (X_centered.T @ U / S).T
                return Vt
        else:
            # Use full SVD for large n_components
            U, S, Vt = svd(X_centered, full_matrices=False)
            return Vt[:n_components]