import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, error_model="numpy", inline="always")
def solve_fast(G, sigma, P_min, S_min):
    n = len(sigma)
    
    # Precompute M and b
    M = np.empty((n, n), dtype=np.float64)
    b = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        G_ii = G[i, i]
        inv_G_ii = 1.0 / G_ii
        factor = S_min * inv_G_ii
        
        b[i] = sigma[i] * factor
        
        for j in range(n):
            M[i, j] = G[i, j] * factor
        M[i, i] = 0.0

    # Gauss-Seidel iteration
    P = np.empty(n, dtype=np.float64)
    for i in range(n):
        P[i] = P_min[i]
        
    tol = 1e-6
    max_iter = 2000 
    
    for _ in range(max_iter):
        max_diff = 0.0
        
        for i in range(n):
            val = b[i]
            for j in range(n):
                val += M[i, j] * P[j]
            
            if val < P_min[i]:
                val = P_min[i]
            
            diff = val - P[i]
            if diff < 0:
                diff = -diff
                
            if diff > max_diff:
                max_diff = diff
            
            P[i] = val
            
        if max_diff < tol:
            break
            
    return P

class Solver:
    def solve(self, problem, **kwargs):
        G = np.asarray(problem["G"], dtype=np.float64)
        sigma = np.asarray(problem["σ"], dtype=np.float64)
        P_min = np.asarray(problem["P_min"], dtype=np.float64)
        S_min = float(problem["S_min"])
        
        P = solve_fast(G, sigma, P_min, S_min)
        
        return {"P": P.tolist()}