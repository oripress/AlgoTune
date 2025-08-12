import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import numba

@numba.njit(fastmath=True, parallel=False)
def sparse_power_method(A, v_init, sparsity_param, max_iter=100, tol=1e-6):
    """Optimized sparse power method with iterative soft thresholding."""
    v = v_init.copy()
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return v
    v /= norm_v
    
    prev_obj = -np.inf
    for it in range(max_iter):
        # Matrix-vector product
        w = A @ v
        
        # Apply soft thresholding
        abs_w = np.abs(w)
        w_soft = np.sign(w) * np.maximum(abs_w - sparsity_param, 0)
        
        # Check for zero vector
        norm_w = np.linalg.norm(w_soft)
        if norm_w < 1e-12:
            return w_soft
            
        w_soft /= norm_w
        
        # Compute objective
        obj = w_soft @ A @ w_soft - sparsity_param * np.sum(np.abs(w_soft))
        
        # Check convergence
        if it > 0 and abs(obj - prev_obj) < tol * (1 + abs(prev_obj)):
            return w_soft
            
        prev_obj = obj
        v = w_soft
        
    return v

class Solver:
    def solve(self, problem, **kwargs) -> dict:
        # Extract problem data
        A = np.array(problem["covariance"])
        n_components = int(problem["n_components"])
        sparsity_param = float(problem["sparsity_param"])
        n = A.shape[0]
        
        if n_components <= 0:
            return {"components": [], "explained_variance": []}
        
        # Initialize components and explained variance
        components = np.zeros((n, n_components))
        explained_variance = np.zeros(n_components)
        
        # Use A directly (symmetric by definition)
        A_sym = A
        
        # Adjust tolerance based on problem size
        tol = 1e-6 if n < 1000 else 1e-4
        
        # Use iterative sparse power method for each component
        for k in range(n_components):
            # Initialize with random vector
            v = np.random.randn(n)
            
            # Compute sparse component
            comp = sparse_power_method(A_sym, v, sparsity_param, max_iter=50, tol=tol)
            
            # Deflate the covariance matrix
            comp_norm = comp @ A_sym @ comp
            if comp_norm > 1e-12:
                A_sym -= comp_norm * np.outer(comp, comp)
            
            # Store results
            components[:, k] = comp
            explained_variance[k] = comp_norm
            
        return {
            "components": components.tolist(),
            "explained_variance": explained_variance.tolist()
        }