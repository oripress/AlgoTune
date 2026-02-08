import numpy as np
from typing import Any
import numba as nb

@nb.njit(cache=True, fastmath=True)
def sinkhorn_core_small(a, b, K, max_iter, tol):
    """Numba Sinkhorn matching POT's sinkhorn_knopp."""
    n = a.shape[0]
    m = b.shape[0]
    u = np.ones(n, dtype=np.float64) / n
    v = np.ones(m, dtype=np.float64) / m
    
    Kv = np.empty(n, dtype=np.float64)
    Ktu = np.empty(m, dtype=np.float64)
    
    for iteration in range(max_iter):
        u_prev = u.copy()
        
        # Ktu = K.T @ u
        for j in range(m):
            s = 0.0
            for i in range(n):
                s += K[i, j] * u[i]
            Ktu[j] = s
        
        v = b / Ktu
        
        # Kv = K @ v
        for i in range(n):
            s = 0.0
            for j in range(m):
                s += K[i, j] * v[j]
            Kv[i] = s
        
        u = a / Kv
        
        # Check convergence like POT
        if iteration % 10 == 0 and iteration > 0:
            # err = max(|u - u_prev|) / max(max(|u|), max(|u_prev|), 1)
            max_diff = 0.0
            max_u = 0.0
            max_uprev = 0.0
            for i in range(n):
                d = abs(u[i] - u_prev[i])
                if d > max_diff:
                    max_diff = d
                au = abs(u[i])
                if au > max_u:
                    max_u = au
                aup = abs(u_prev[i])
                if aup > max_uprev:
                    max_uprev = aup
            denom = max_u
            if max_uprev > denom:
                denom = max_uprev
            if 1.0 > denom:
                denom = 1.0
            err = max_diff / denom
            if err < tol:
                break
    
    # Build G
    G = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            G[i, j] = u[i] * K[i, j] * v[j]
    
    return G


@nb.njit(cache=True, fastmath=True)
def sinkhorn_log_core(a, b, M, reg, max_iter, tol):
    """Log-domain Sinkhorn for numerical stability."""
    n = a.shape[0]
    m = b.shape[0]
    
    # log-domain dual variables (log(u), log(v))
    log_u = np.zeros(n, dtype=np.float64)
    log_v = np.zeros(m, dtype=np.float64)
    log_a = np.log(a)
    log_b = np.log(b)
    
    for iteration in range(max_iter):
        log_u_prev = log_u.copy()
        
        # Update log_v: log_v_j = log(b_j) - logsumexp_i(log_u_i - M_ij/reg)
        for j in range(m):
            max_val = -1e300
            for i in range(n):
                val = log_u[i] - M[i, j] / reg
                if val > max_val:
                    max_val = val
            s = 0.0
            for i in range(n):
                s += np.exp(log_u[i] - M[i, j] / reg - max_val)
            log_v[j] = log_b[j] - max_val - np.log(s)
        
        # Update log_u: log_u_i = log(a_i) - logsumexp_j(log_v_j - M_ij/reg)
        for i in range(n):
            max_val = -1e300
            for j in range(m):
                val = log_v[j] - M[i, j] / reg
                if val > max_val:
                    max_val = val
            s = 0.0
            for j in range(m):
                s += np.exp(log_v[j] - M[i, j] / reg - max_val)
            log_u[i] = log_a[i] - max_val - np.log(s)
        
        # Check convergence
        if iteration % 10 == 0 and iteration > 0:
            max_diff = 0.0
            max_lu = 0.0
            max_luprev = 0.0
            for i in range(n):
                d = abs(log_u[i] - log_u_prev[i])
                if d > max_diff:
                    max_diff = d
                au = abs(log_u[i])
                if au > max_lu:
                    max_lu = au
                aup = abs(log_u_prev[i])
                if aup > max_luprev:
                    max_luprev = aup
            denom = max_lu
            if max_luprev > denom:
                denom = max_luprev
            if 1.0 > denom:
                denom = 1.0
            err = max_diff / denom
            if err < tol:
                break
    
    # Compute transport plan: G_ij = exp(log_u_i + log_v_j - M_ij/reg)
    G = np.empty((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            G[i, j] = np.exp(log_u[i] + log_v[j] - M[i, j] / reg)
    
    return G


class Solver:
    def __init__(self):
        # Warm up numba
        a_dummy = np.array([0.5, 0.5])
        b_dummy = np.array([0.5, 0.5])
        M_dummy = np.array([[0.0, 1.0], [1.0, 0.0]])
        K_dummy = np.exp(-M_dummy / 1.0)
        sinkhorn_core_small(a_dummy, b_dummy, K_dummy, 10, 1e-9)
        sinkhorn_log_core(a_dummy, b_dummy, M_dummy, 1.0, 10, 1e-9)
    
    def solve(self, problem: dict, **kwargs) -> Any:
        a = np.array(problem["source_weights"], dtype=np.float64)
        b = np.array(problem["target_weights"], dtype=np.float64)
        M = np.ascontiguousarray(problem["cost_matrix"], dtype=np.float64)
        reg = float(problem["reg"])
        
        n, m = len(a), len(b)
        tol = 1e-9
        max_iter = 1000
        
        # For large problems, use numpy BLAS-based approach
        if n * m > 62500:  # ~250x250
            K = np.exp(-M / reg)
            if not np.all(np.isfinite(K)) or np.min(K) == 0:
                G = sinkhorn_log_core(a, b, M, reg, max_iter, tol)
            else:
                u = np.ones(n, dtype=np.float64) / n
                KT = np.ascontiguousarray(K.T)
                
                for it in range(max_iter):
                    u_prev = u
                    v = b / (KT @ u)
                    Kv = K @ v
                    u = a / Kv
                    
                    if it % 10 == 0 and it > 0:
                        max_diff = np.max(np.abs(u - u_prev))
                        denom = max(np.max(np.abs(u)), np.max(np.abs(u_prev)), 1.0)
                        err = max_diff / denom
                        if err < tol:
                            break
                
                G = u[:, None] * K * v[None, :]
                
                if not np.isfinite(G).all():
                    G = sinkhorn_log_core(a, b, M, reg, max_iter, tol)
        else:
            K = np.exp(-M / reg)
            if np.min(K) < 1e-300 or not np.all(np.isfinite(K)):
                G = sinkhorn_log_core(a, b, M, reg, max_iter, tol)
            else:
                G = sinkhorn_core_small(a, b, K, max_iter, tol)
                if not np.isfinite(G).all():
                    G = sinkhorn_log_core(a, b, M, reg, max_iter, tol)
        
        if not np.isfinite(G).all():
            return {"transport_plan": None, "error_message": "Non-finite values"}
        
        return {"transport_plan": G, "error_message": None}