import numpy as np
from scipy.optimize import linprog
from numba import njit

@njit(cache=True)
def foschini_miljanic(off_diag, diag_G, σ, P_min, P_max, S_min, n, max_iter=300):
    P = P_min.copy()
    for _ in range(max_iter):
        interference = np.dot(off_diag, P) + σ
        P_new = S_min * interference / diag_G
        for i in range(n):
            if P_new[i] < P_min[i]:
                P_new[i] = P_min[i]
            elif P_new[i] > P_max[i]:
                P_new[i] = P_max[i]
        converged = True
        for i in range(n):
            if abs(P_new[i] - P[i]) > 1e-11 + 1e-9 * abs(P[i]):
                converged = False
                break
        if converged:
            sinr_ok = True
            for i in range(n):
                interf = σ[i] + np.dot(off_diag[i], P_new)
                if diag_G[i] * P_new[i] < (S_min - 1e-9) * interf:
                    sinr_ok = False
                    break
            return P_new, sinr_ok
        P = P_new
    return P, False

class Solver:
    def solve(self, problem, **kwargs):
        G = np.asarray(problem["G"], dtype=np.float64)
        σ = np.asarray(problem["σ"], dtype=np.float64)
        P_min = np.asarray(problem["P_min"], dtype=np.float64)
        P_max = np.asarray(problem["P_max"], dtype=np.float64)
        S_min = float(problem["S_min"])
        n = G.shape[0]
        
        diag_G = np.diag(G).copy()
        off_diag = G.copy()
        off_diag.flat[::n+1] = 0
        
        # Try FM iteration first (fastest)
        P, success = foschini_miljanic(off_diag, diag_G, σ, P_min, P_max, S_min, n)
        if success:
            return {"P": P.tolist(), "objective": float(np.sum(P))}
        
        # Build constraint matrix for direct solve / LP
        A = -S_min * G
        A.flat[::n+1] = diag_G
        b = S_min * σ
        
        # Try direct solution
        try:
            P = np.linalg.solve(A, b)
            if np.all(P >= P_min - 1e-9) and np.all(P <= P_max + 1e-9):
                P = np.clip(P, P_min, P_max)
                return {"P": P.tolist(), "objective": float(np.sum(P))}
        except np.linalg.LinAlgError:
            pass
        
        # LP fallback
        result = linprog(np.ones(n), A_ub=-A, b_ub=-b, bounds=list(zip(P_min, P_max)), method='highs')
        if not result.success:
            raise ValueError(f"Solver failed: {result.message}")
        return {"P": result.x.tolist(), "objective": result.fun}