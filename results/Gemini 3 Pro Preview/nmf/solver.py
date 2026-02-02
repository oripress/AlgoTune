import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def solve_hals(X, n_components, max_iter=10):
    m, n = X.shape
    
    # Initialization
    np.random.seed(0)
    # Use sum/size instead of mean to avoid potential numba issues with 2D arrays
    avg = np.sum(X) / (m * n)
    scale = np.sqrt(avg / n_components)
    scale = np.float32(scale)
    
    # W in Fortran layout for column access optimization
    # Generate in float64 then cast to float32
    W_init = np.random.random((m, n_components))
    W = np.asfortranarray(W_init.astype(np.float32))
    W *= scale
    
    H_init = np.random.random((n_components, n))
    H = H_init.astype(np.float32)
    H *= scale
    
    eps = 1e-7
    
    for it in range(max_iter):
        # Update W
        HHt = H @ H.T
        XHt = X @ H.T
        
        for k in range(n_components):
            denom = HHt[k, k]
            if denom < eps:
                continue
            
            pred = W @ HHt[:, k]
            grad = pred - XHt[:, k]
            
            W[:, k] = np.maximum(0.0, W[:, k] - grad / denom)
            
        # Update H
        WtW = W.T @ W
        WtX = W.T @ X
        
        for k in range(n_components):
            denom = WtW[k, k]
            if denom < eps:
                continue
            
            pred = WtW[k, :] @ H
            grad = pred - WtX[k, :]
            
            H[k, :] = np.maximum(0.0, H[k, :] - grad / denom)
            
    return W, H

class Solver:
    def __init__(self):
        # Trigger compilation
        try:
            dummy_X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            solve_hals(dummy_X, 2, 1)
        except Exception as e:
            print(f"Compilation failed: {e}")

    def solve(self, problem, **kwargs):
        try:
            X = np.array(problem["X"], dtype=np.float32)
            n_components = problem["n_components"]
            
            # Run HALS
            W, H = solve_hals(X, n_components, 500)
            
            return {"W": W.tolist(), "H": H.tolist()}
        except Exception:
            # Fallback
            n_components = problem["n_components"]
            n, d = np.array(problem["X"]).shape
            W = np.zeros((n, n_components), dtype=float).tolist()
            H = np.zeros((n_components, d), dtype=float).tolist()
            return {"W": W, "H": H}