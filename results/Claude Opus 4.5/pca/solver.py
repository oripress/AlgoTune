import numpy as np
from scipy.linalg import svd as scipy_svd, eigh
from sklearn.utils.extmath import randomized_svd

class Solver:
    def solve(self, problem, **kwargs):
        X = np.array(problem["X"], dtype=np.float64)
        n_components = problem["n_components"]
        m, n = X.shape
        
        # Center the data
        X -= np.mean(X, axis=0)
        
        min_dim = min(m, n)
        
        # For very small matrices or when we need almost all components
        if min_dim <= 10 or n_components >= min_dim - 1:
            _, _, Vt = scipy_svd(X, full_matrices=False, check_finite=False)
            return Vt[:n_components]
        
        # If n_components is small relative to dimensions, use randomized SVD
        if n_components <= min(m, n) * 0.2 and min_dim > 50:
            _, _, Vt = randomized_svd(X, n_components=n_components, n_iter=4, random_state=42)
            return Vt
        
        # For moderate cases, use eigendecomposition on smaller matrix
        if m < n:
            # Compute eigendecomposition of X @ X.T (m x m matrix)
            C = X @ X.T
            eigenvalues, U = eigh(C, check_finite=False)
            # eigh returns eigenvalues in ascending order, we want descending
            U = U[:, ::-1][:, :n_components]
            # V = X.T @ U normalized
            Vt = U.T @ X
            # Normalize rows
            norms = np.linalg.norm(Vt, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            Vt = Vt / norms
            return Vt
        else:
            # Compute eigendecomposition of X.T @ X (n x n matrix)
            C = X.T @ X
            eigenvalues, V = eigh(C, check_finite=False)
            # eigh returns eigenvalues in ascending order, we want descending
            Vt = V[:, ::-1][:, :n_components].T
            return Vt