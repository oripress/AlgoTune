import numpy as np
from numba import njit

@njit(fastmath=True, cache=True)
def solve_ba(P, P_T, max_iter=10000, tol=1e-10):
    m, n = P.shape
    
    # Precompute c
    c = np.zeros(n)
    for j in range(n):
        val = 0.0
        for i in range(m):
            p_val = P[i, j]
            if p_val > 1e-20:
                val += p_val * np.log(p_val)
        c[j] = val
        
    x = np.full(n, 1.0/n)
    
    for k in range(max_iter):
        # y = P @ x
        y = np.dot(P, x)
        
        # log_y
        log_y = np.log(y + 1e-300)
        
        # d = c - P.T @ log_y
        term = np.dot(P_T, log_y)
        d = c - term
        
        max_d = -1e300
        for i in range(n):
            if d[i] > max_d:
                max_d = d[i]
        
        mi = 0.0
        for i in range(n):
            mi += x[i] * d[i]
            
        if max_d - mi < tol:
            break
            
        # Update x
        sum_x = 0.0
        for i in range(n):
            x[i] = x[i] * np.exp(d[i] - max_d)
            sum_x += x[i]
            
        for i in range(n):
            x[i] /= sum_x
            
    # Final C
    y = np.dot(P, x)
    log_y = np.log(y + 1e-300)
    term = np.dot(P_T, log_y)
    d = c - term
    C_nat = 0.0
    for i in range(n):
        C_nat += x[i] * d[i]
        
    return x, C_nat

class Solver:
    def solve(self, problem, **kwargs):
        if problem is None:
            return None
        P_list = problem.get("P")
        if P_list is None:
            return None
            
        try:
            P_input = np.array(P_list, dtype=np.float64)
        except Exception:
            return None
            
        if P_input.ndim != 2:
            return None
        m, n = P_input.shape
        if m == 0 or n == 0:
            return None
            
        # Check column sums
        col_sums = P_input.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=1e-5):
            return None
            
        # Remove zero rows
        row_sums = P_input.sum(axis=1)
        active_rows = row_sums > 1e-15
        P = P_input[active_rows]
        
        # If all rows removed (shouldn't happen if cols sum to 1), return None
        if P.shape[0] == 0:
            return None
            
        P = np.ascontiguousarray(P)
        P_T = np.ascontiguousarray(P.T)
        
        x, C_nat = solve_ba(P, P_T)
        
        # Ensure x is valid
        if np.any(np.isnan(x)):
            return None
            
        return {"x": x.tolist(), "C": C_nat / np.log(2.0)}