from typing import Any
import numpy as np
from scipy.linalg import svd, eigh

class Solver:
    def solve(self, problem: dict[str, Any]) -> Any:
        """
        Principal component analysis (PCA) using a fast, shape-aware SciPy algorithm.

        This solver determines the most efficient method based on the dimensions of
        the input data matrix X (n_samples vs n_features).

        - If n_samples >= n_features (tall or square matrix):
          It computes the covariance matrix and uses `scipy.linalg.eigh`. This is
          faster as it works on a smaller (n_features x n_features) matrix.

        - If n_samples < n_features (wide matrix):
          It uses `scipy.linalg.svd` on the centered data matrix. This avoids
          forming the large (n_features x n_features) covariance matrix, which
          would be computationally expensive.
        """
        n_components = problem["n_components"]
        X_list = problem["X"]

        if not X_list or not X_list[0] or n_components == 0:
            return []

        try:
            # Convert to NumPy array for efficient computation
            X = np.array(X_list, dtype=np.float64)
            n_samples, n_features = X.shape

            # Handle edge case where n_components is invalid
            if n_components > n_features:
                # PCA components cannot exceed the number of features.
                # Returning an empty list as a signal of invalid input.
                return []

            # Center the data by subtracting the mean of each feature
            X_centered = X - X.mean(axis=0)

            # Choose the most efficient SVD/eigen-decomposition method
            if n_samples >= n_features:
                # Standard PCA: Eigendecomposition of the covariance matrix
                # This is faster when n_features is small.
                # C = (X_centered^T @ X_centered) / (n_samples - 1)
                # We can ignore the scaling factor as it doesn't affect eigenvectors.
                scatter_matrix = X_centered.T @ X_centered
                
                # Use eigh for symmetric matrices; it's faster than eig.
                # It returns eigenvalues in ascending order.
                _, vecs = eigh(scatter_matrix)
                
                # Components are the eigenvectors corresponding to the largest eigenvalues.
                components = vecs[:, -n_components:]
                
                # eigh returns eigenvalues in ascending order, so we reverse the
                # order of eigenvectors to get descending principal components.
                return components.T[::-1].tolist()
            else:
                # SVD on the data matrix is faster when n_features > n_samples.
                # X = U @ S @ Vh. The principal components are the rows of Vh.
                _, _, vh = svd(X_centered, full_matrices=False)
                
                return vh[:n_components, :].tolist()

        except (ValueError, IndexError):
            # Fallback for malformed input (e.g., inconsistent row lengths)
            return []