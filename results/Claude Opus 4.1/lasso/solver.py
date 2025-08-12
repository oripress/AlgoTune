import numpy as np
from numba import jit, prange
from typing import Any

class Solver:
    def __init__(self):
        # Pre-compile the coordinate descent function
        self._coordinate_descent = coordinate_descent_lasso
    
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        X = np.array(problem["X"], dtype=np.float64)
        y = np.array(problem["y"], dtype=np.float64)
        
        n, d = X.shape
        alpha = 0.1
        
        # Initialize weights
        w = np.zeros(d, dtype=np.float64)
        
        # Run coordinate descent
        w = self._coordinate_descent(X, y, w, alpha, n, d)
        
        return w.tolist()

@jit(nopython=True, cache=True, fastmath=True)
def coordinate_descent_lasso(X, y, w, alpha, n, d, max_iter=1000, tol=1e-4):
    """
    Fast coordinate descent for Lasso regression.
    """
    # Precompute X^T X diagonal and X^T y for efficiency
    XTX_diag = np.zeros(d)
    for j in range(d):
        for i in range(n):
            XTX_diag[j] += X[i, j] * X[i, j]
    
    XTy = np.zeros(d)
    for j in range(d):
        for i in range(n):
            XTy[j] += X[i, j] * y[i]
    
    # Residuals
    residuals = y.copy()
    for i in range(n):
        for j in range(d):
            residuals[i] -= X[i, j] * w[j]
    
    # Coordinate descent iterations
    for iteration in range(max_iter):
        w_old = w.copy()
        
        for j in range(d):
            # Skip if feature has zero variance
            if XTX_diag[j] == 0:
                continue
            
            # Add back the contribution of w[j] to residuals
            for i in range(n):
                residuals[i] += X[i, j] * w[j]
            
            # Compute the gradient for coordinate j
            rho = 0.0
            for i in range(n):
                rho += X[i, j] * residuals[i]
            rho /= n
            
            # Soft thresholding
            if rho > alpha:
                w[j] = (rho - alpha) / (XTX_diag[j] / n)
            elif rho < -alpha:
                w[j] = (rho + alpha) / (XTX_diag[j] / n)
            else:
                w[j] = 0.0
            
            # Update residuals with new w[j]
            for i in range(n):
                residuals[i] -= X[i, j] * w[j]
        
        # Check convergence
        converged = True
        for j in range(d):
            if abs(w[j] - w_old[j]) > tol:
                converged = False
                break
        
        if converged:
            break
    
    return w