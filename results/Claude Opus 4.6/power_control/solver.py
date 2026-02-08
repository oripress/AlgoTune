import numpy as np
from scipy.optimize import linprog

class Solver:
    def solve(self, problem, **kwargs):
        G = np.asarray(problem["G"], dtype=np.float64)
        sigma = np.asarray(problem["σ"], dtype=np.float64)
        P_min = np.asarray(problem["P_min"], dtype=np.float64)
        P_max = np.asarray(problem["P_max"], dtype=np.float64)
        S_min = float(problem["S_min"])
        n = G.shape[0]
        
        d = np.diag(G).copy()
        inv_d = np.reciprocal(d)
        
        # M = I - F where F[i,j] = S_min*G[i,j]/G[i,i] for j!=i
        M = -(S_min * (G * inv_d[:, None]))
        np.fill_diagonal(M, 1.0)
        b = (S_min * sigma) * inv_d
        
        # Try all SINR constraints active
        try:
            P = np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            return self._lp(G, sigma, P_min, P_max, S_min, n)
        
        if np.all(P >= P_min - 1e-10) and np.all(P <= P_max + 1e-10):
            np.clip(P, P_min, P_max, out=P)
            return {"P": P.tolist(), "objective": float(P.sum())}
        
        # Iterative fixing of bound-violating variables
        fixed = np.zeros(n, dtype=bool)
        P_fix = np.empty(n)
        
        for _ in range(n):
            lo = (P < P_min - 1e-10) & ~fixed
            hi = (P > P_max + 1e-10) & ~fixed
            new_fix = lo | hi
            if not new_fix.any():
                break
            fixed |= new_fix
            P_fix[lo] = P_min[lo]
            P_fix[hi] = P_max[hi]
            free = ~fixed
            if not free.any():
                P[:] = P_fix
                break
            fi = np.where(fixed)[0]
            fr = np.where(free)[0]
            rhs = b[fr] - M[np.ix_(fr, fi)] @ P_fix[fi]
            try:
                P_free = np.linalg.solve(M[np.ix_(fr, fr)], rhs)
            except np.linalg.LinAlgError:
                return self._lp(G, sigma, P_min, P_max, S_min, n)
            P[fr] = P_free
            P[fi] = P_fix[fi]
        
        if np.all(P >= P_min - 1e-10) and np.all(P <= P_max + 1e-10):
            np.clip(P, P_min, P_max, out=P)
            return {"P": P.tolist(), "objective": float(P.sum())}
        
        return self._lp(G, sigma, P_min, P_max, S_min, n)
    
    def _lp(self, G, sigma, P_min, P_max, S_min, n):
        c = np.ones(n)
        A_ub = S_min * G.copy()
        idx = np.arange(n)
        A_ub[idx, idx] = -np.diag(G)
        b_ub = -S_min * sigma
        bounds = list(zip(P_min.tolist(), P_max.tolist()))
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not result.success:
            raise ValueError(f"Solver failed: {result.message}")
        return {"P": result.x.tolist(), "objective": float(result.fun)}