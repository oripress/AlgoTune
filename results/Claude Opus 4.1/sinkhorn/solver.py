import numpy as np
from typing import Any
import numba

@numba.jit(nopython=True, cache=True, fastmath=True)
def sinkhorn_fast(a, b, M, reg, max_iter=100, tol=1e-8):
    """Fast Sinkhorn algorithm with optimized operations."""
    n, m = M.shape
    
    # Compute kernel matrix K = exp(-M/reg)
    K = np.exp(-M / reg)
    
    # Initialize dual variables
    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)
    
    # Pre-allocate arrays for efficiency
    Kv = np.empty(n, dtype=np.float64)
    KTu = np.empty(m, dtype=np.float64)
    
    for iteration in range(max_iter):
        # Store previous u for convergence check
        u_prev = u.copy()
        
        # Compute Kv = K @ v efficiently
        for i in range(n):
            sum_val = 0.0
            for j in range(m):
                sum_val += K[i, j] * v[j]
            Kv[i] = sum_val
        
        # Update u = a / Kv
        for i in range(n):
            u[i] = a[i] / (Kv[i] + 1e-16)
        
        # Compute KTu = K.T @ u efficiently
        for j in range(m):
            sum_val = 0.0
            for i in range(n):
                sum_val += K[i, j] * u[i]
            KTu[j] = sum_val
        
        # Update v = b / KTu
        for j in range(m):
            v[j] = b[j] / (KTu[j] + 1e-16)
        
        # Check convergence using relative change
        max_diff = 0.0
        for i in range(n):
            diff = abs(u[i] - u_prev[i]) / (abs(u[i]) + 1e-16)
            if diff > max_diff:
                max_diff = diff
        
        if max_diff < tol:
            break
    
    # Compute transport plan G = diag(u) @ K @ diag(v)
    G = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            G[i, j] = u[i] * K[i, j] * v[j]
    
    return G

# Compile the function ahead of time with a dummy call
@numba.jit(nopython=True, cache=True)
def warmup_jit():
    """Warmup function to trigger JIT compilation."""
    a = np.array([0.5, 0.5], dtype=np.float64)
    b = np.array([0.5, 0.5], dtype=np.float64)
    M = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    reg = 1.0
    return sinkhorn_fast(a, b, M, reg, max_iter=10)

class Solver:
    def __init__(self):
        # Trigger JIT compilation during initialization
        try:
            _ = warmup_jit()
        except:
            pass
    
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Solve entropic optimal transport using optimized Sinkhorn algorithm."""
        # Extract input data
        a = np.ascontiguousarray(problem["source_weights"], dtype=np.float64)
        b = np.ascontiguousarray(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        
        try:
            G = sinkhorn_fast(a, b, M, reg)
            
            # Check for non-finite values
            if not np.isfinite(G).all():
                raise ValueError("Non-finite values in transport plan")
            
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}