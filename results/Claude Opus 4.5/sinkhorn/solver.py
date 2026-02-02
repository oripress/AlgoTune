import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _compute_G(u, K, v, n, m):
    G = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            G[i, j] = u[i] * K[i, j] * v[j]
    return G

def _sinkhorn_numpy(a, b, K, max_iter, tol):
    n = a.shape[0]
    m = b.shape[0]
    
    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)
    
    for iteration in range(max_iter):
        u_old = u.copy()
        
        # v = b / (K^T @ u)
        Ktu = K.T @ u
        np.divide(b, Ktu, out=v, where=Ktu > 1e-300)
        
        # u = a / (K @ v)
        Kv = K @ v
        np.divide(a, Kv, out=u, where=Kv > 1e-300)
        
        # Check convergence
        err = np.max(np.abs(u - u_old))
        if err < tol:
            break
    
    # Compute transport plan G = diag(u) @ K @ diag(v)
    G = _compute_G(u, K, v, n, m)
    return G

class Solver:
    def __init__(self):
        # Warm up JIT
        a = np.array([0.5, 0.5], dtype=np.float64)
        b = np.array([0.5, 0.5], dtype=np.float64)
        K = np.array([[1.0, 0.368], [0.368, 1.0]], dtype=np.float64)
        _compute_G(np.ones(2), K, np.ones(2), 2, 2)
    
    def solve(self, problem, **kwargs):
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        
        try:
            # Compute K = exp(-M/reg)
            K = np.exp(-M / reg)
            
            G = _sinkhorn_numpy(a, b, K, 1000, 1e-9)
            
            if not np.isfinite(G).all():
                raise ValueError("Non-finite values in transport plan")
            return {"transport_plan": G, "error_message": None}
        except Exception as exc:
            return {"transport_plan": None, "error_message": str(exc)}