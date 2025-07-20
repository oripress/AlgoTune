import numpy as np
from typing import Any
import numba

@numba.jit(nopython=True, fastmath=True, cache=True)
def coordinate_descent_lasso(X, y, alpha, max_iter=200, tol=1e-5):
    """Fast coordinate descent for Lasso using numba."""
    n, d = X.shape
    w = np.zeros(d)
    
    # Precompute column norms
    col_norms = np.zeros(d)
    for j in range(d):
        col_norms[j] = np.dot(X[:, j], X[:, j]) / n
    
    # Initialize residual
    r = y.copy()
    
    for _ in range(max_iter):
        max_change = 0.0
        
        for j in range(d):
            if col_norms[j] == 0:
                continue
            
            # Store old value
            w_j_old = w[j]
            
            # Add back contribution
            if w_j_old != 0:
                for i in range(n):
                    r[i] += X[i, j] * w_j_old
            
            # Compute gradient
            grad = 0.0
            for i in range(n):
                grad += X[i, j] * r[i]
            grad /= n
            
            # Soft thresholding
            if grad > alpha:
                w[j] = (grad - alpha) / col_norms[j]
            elif grad < -alpha:
                w[j] = (grad + alpha) / col_norms[j]
            else:
                w[j] = 0.0
            
            # Update residual
            if w[j] != 0:
                for i in range(n):
                    r[i] -= X[i, j] * w[j]
            
            # Track maximum change
            change = abs(w[j] - w_j_old)
            if change > max_change:
                max_change = change
        
        # Check convergence
        if max_change < tol:
            break
    
    return w

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        """Solve Lasso regression using numba-optimized coordinate descent."""
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        
        # Use the JIT-compiled function
        w = coordinate_descent_lasso(X, y, 0.1)
        
        return w.tolist()