import numpy as np
from numba import jit

@jit(nopython=True, cache=True, fastmath=True)
def coordinate_descent_solver(b, sparsity_param, max_iter=50):
    """Solve using coordinate descent with soft-thresholding"""
    n = len(b)
    x = b.copy()
    
    # Initialize with scaled b
    norm_sq = 0.0
    for i in range(n):
        norm_sq += x[i] * x[i]
    if norm_sq > 1.0:
        scale = 1.0 / np.sqrt(norm_sq)
        for i in range(n):
            x[i] *= scale
    
    tol = 1e-8
    
    for iteration in range(max_iter):
        x_max_change = 0.0
        
        for j in range(n):
            old_xj = x[j]
            
            # Compute residual without x[j]
            # For ||b - x||^2 + λ||x||_1, the update for x[j] is:
            # x[j] = soft_threshold(b[j], λ/2)
            
            # Soft-thresholding
            if b[j] > sparsity_param / 2.0:
                x[j] = b[j] - sparsity_param / 2.0
            elif b[j] < -sparsity_param / 2.0:
                x[j] = b[j] + sparsity_param / 2.0
            else:
                x[j] = 0.0
            
            # Check for change
            change = abs(x[j] - old_xj)
            if change > x_max_change:
                x_max_change = change
        
        # Project to unit ball
        norm_sq = 0.0
        for i in range(n):
            norm_sq += x[i] * x[i]
        if norm_sq > 1.0:
            scale = 1.0 / np.sqrt(norm_sq)
            for i in range(n):
                x[i] *= scale
        
        if x_max_change < tol:
            break
    
    return x

@jit(nopython=True, cache=True, fastmath=True)
def proximal_gradient_solver(b, sparsity_param, max_iter=80):
    """Fast proximal gradient with adaptive restart"""
    n = len(b)
    x = b.copy()
    y = b.copy()
    x_old = np.empty(n)
    
    # Initialize
    norm_sq = 0.0
    for i in range(n):
        norm_sq += x[i] * x[i]
    if norm_sq > 1.0:
        scale = 1.0 / np.sqrt(norm_sq)
        for i in range(n):
            x[i] *= scale
            y[i] = x[i]
    
    step_size = 0.5
    threshold = sparsity_param * step_size
    tol_sq = 1e-12
    t = 1.0
    
    for iteration in range(max_iter):
        # Save old x
        for i in range(n):
            x_old[i] = x[i]
        
        # Gradient step: gradient of ||b-x||^2 is 2(x-b)
        for i in range(n):
            grad = 2.0 * (y[i] - b[i])
            x[i] = y[i] - step_size * grad
        
        # Soft-thresholding
        for i in range(n):
            if x[i] > threshold:
                x[i] -= threshold
            elif x[i] < -threshold:
                x[i] += threshold
            else:
                x[i] = 0.0
        
        # Project to unit ball
        norm_sq = 0.0
        for i in range(n):
            norm_sq += x[i] * x[i]
        if norm_sq > 1.0:
            scale = 1.0 / np.sqrt(norm_sq)
            for i in range(n):
                x[i] *= scale
        
        # Check convergence
        diff_sq = 0.0
        for i in range(n):
            diff = x[i] - x_old[i]
            diff_sq += diff * diff
        
        if diff_sq < tol_sq:
            break
        
        # FISTA momentum
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        beta = (t - 1.0) / t_new
        
        for i in range(n):
            y[i] = x[i] + beta * (x[i] - x_old[i])
        
        t = t_new
    
    return x

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the sparse PCA problem.
        
        :param problem: Dictionary with problem parameters
        :return: Dictionary with the sparse principal components
        """
        A = np.array(problem["covariance"], dtype=np.float64)
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        
        # Get eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(A)
        
        # Keep only positive eigenvalues
        pos_indices = eigvals > 0
        if not np.any(pos_indices):
            return {"components": np.zeros((A.shape[0], n_components)).tolist(),
                    "explained_variance": [0.0] * n_components}
        
        eigvals = eigvals[pos_indices]
        eigvecs = eigvecs[:, pos_indices]
        
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        k = min(len(eigvals), n_components)
        B = eigvecs[:, :k] * np.sqrt(eigvals[:k])
        
        # Solve for each component using the faster method based on sparsity
        X = np.empty((A.shape[0], k), dtype=np.float64)
        for i in range(k):
            # Use coordinate descent for high sparsity, proximal gradient otherwise
            if sparsity_param > 0.5:
                X[:, i] = coordinate_descent_solver(B[:, i], sparsity_param)
            else:
                X[:, i] = proximal_gradient_solver(B[:, i], sparsity_param)
        
        # Calculate explained variance
        explained_variance = []
        for i in range(k):
            var = float(X[:, i] @ A @ X[:, i])
            explained_variance.append(var)
        
        return {"components": X.tolist(), "explained_variance": explained_variance}