import numpy as np
from typing import Any, Dict
import numba as nb

@nb.njit(fastmath=True)
def soft_threshold(x, threshold):
    """Fast soft thresholding for L1 proximal operator."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)

@nb.njit(fastmath=True)
def project_unit_ball_column(x):
    """Project a vector onto unit ball."""
    norm = np.linalg.norm(x)
    if norm > 1.0:
        return x / norm
    return x

@nb.njit(fastmath=True)
def proximal_gradient_descent(B, A, n_components, sparsity_param):
    """
    Fast proximal gradient descent for sparse PCA.
    """
    n = B.shape[0]
    X = B.copy()
    
    # Adaptive step size using Lipschitz constant
    # The Lipschitz constant for gradient of ||B-X||^2 is 2
    step_size = 0.5
    
    max_iter = 50  # Reduced iterations since we converge faster
    momentum = 0.9
    
    # Momentum acceleration
    X_prev = X.copy()
    Y = X.copy()
    
    for iteration in range(max_iter):
        # Gradient of smooth part: 2*(Y - B)
        grad = 2.0 * (Y - B)
        
        # Gradient descent step
        X_new = Y - step_size * grad
        
        # Soft thresholding for L1 penalty
        threshold = step_size * sparsity_param
        for i in range(n):
            for j in range(n_components):
                X_new[i, j] = soft_threshold(X_new[i, j], threshold)
        
        # Project each column onto unit ball
        for j in range(n_components):
            X_new[:, j] = project_unit_ball_column(X_new[:, j])
        
        # Momentum update (FISTA acceleration)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum)) / 2.0
        Y = X_new + ((momentum - 1.0) / t_new) * (X_new - X_prev)
        
        # Check convergence (simplified for speed)
        if iteration > 5:
            diff = np.sum(np.abs(X_new - X_prev))
            if diff < 1e-5:
                break
        
        X_prev = X.copy()
        X = X_new
        momentum = t_new
    
    return X

class Solver:
    def __init__(self):
        """Initialize the solver."""
        pass
    
    def solve(self, problem: Dict, **kwargs) -> Any:
        """
        Solve the sparse PCA problem using fast proximal gradient descent.
        """
        A = np.asarray(problem["covariance"], dtype=np.float64)
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        
        n = A.shape[0]
        
        # Fast eigendecomposition - use eigh which is optimized for symmetric matrices
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Keep only positive eigenvalues (vectorized)
        pos_mask = eigvals > 0
        if not np.any(pos_mask):
            return {"components": [], "explained_variance": []}
        
        eigvals = eigvals[pos_mask]
        eigvecs = eigvecs[:, pos_mask]
        
        # Sort in descending order (get indices once)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Create B matrix
        k = min(len(eigvals), n_components)
        if k == 0:
            return {"components": [], "explained_variance": []}
        
        # Vectorized B computation
        B = eigvecs[:, :k] * np.sqrt(eigvals[:k])
        
        # Run optimized proximal gradient descent
        X = proximal_gradient_descent(B, A, n_components, sparsity_param)
        
        # Calculate explained variance (vectorized)
        explained_variance = np.diag(X.T @ A @ X).tolist()
        
        return {
            "components": X.tolist(),
            "explained_variance": explained_variance
        }