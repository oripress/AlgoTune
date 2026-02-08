import numpy as np
import numba as nb

@nb.njit(cache=True)
def _dare_sda_numba(A, G, n):
    """Solve DARE using SDA with Q=I, R=I, fully in numba."""
    Ak = A.copy()
    Gk = G.copy()
    Hk = np.eye(n)
    In = np.eye(n)
    
    for it in range(80):
        M = In + Gk @ Hk
        Phi = np.linalg.solve(M, In)
        PhiAk = Phi @ Ak
        
        Hk_new = Hk + Ak.T @ (Hk @ PhiAk)
        Gk = Gk + Ak @ (Phi @ (Gk @ Ak.T))
        Ak = Ak @ PhiAk
        
        diff = 0.0
        hn = 0.0
        for i in range(n):
            for j in range(n):
                d = Hk_new[i, j] - Hk[i, j]
                diff += d * d
                hn += Hk_new[i, j] * Hk_new[i, j]
        
        Hk = Hk_new
        
        if diff < 1e-28 * (hn + 1.0):
            break
    
    # Symmetrize
    for i in range(n):
        for j in range(i+1, n):
            v = 0.5 * (Hk[i, j] + Hk[j, i])
            Hk[i, j] = v
            Hk[j, i] = v
    
    return Hk

@nb.njit(cache=True)
def _solve_all(A, B, n, m):
    """Solve DARE and compute K, check validity. Returns (success, K, P)."""
    G = B @ B.T
    
    # SDA
    Ak = A.copy()
    Gk = G.copy()
    Hk = np.eye(n)
    In = np.eye(n)
    
    converged = False
    for it in range(80):
        M = In + Gk @ Hk
        Phi = np.linalg.solve(M, In)
        PhiAk = Phi @ Ak
        
        Hk_new = Hk + Ak.T @ (Hk @ PhiAk)
        Gk = Gk + Ak @ (Phi @ (Gk @ Ak.T))
        Ak = Ak @ PhiAk
        
        diff = 0.0
        hn = 0.0
        for i in range(n):
            for j in range(n):
                d = Hk_new[i, j] - Hk[i, j]
                diff += d * d
                hn += Hk_new[i, j] * Hk_new[i, j]
        
        Hk = Hk_new
        
        if diff < 1e-28 * (hn + 1.0):
            converged = True
            break
    
    if not converged:
        return False, np.empty((m, n)), np.empty((n, n))
    
    # Symmetrize P
    P = Hk
    for i in range(n):
        for j in range(i+1, n):
            v = 0.5 * (P[i, j] + P[j, i])
            P[i, j] = v
            P[j, i] = v
    
    # Check P positive definite: min diagonal > 0
    for i in range(n):
        if P[i, i] <= 1e-10:
            return False, np.empty((m, n)), np.empty((n, n))
    
    # K = -(B^T P B + I)^{-1} B^T P A
    BtP = B.T @ P
    BtPB_R = BtP @ B
    for i in range(m):
        BtPB_R[i, i] += 1.0
    BtPA = BtP @ A
    K = -np.linalg.solve(BtPB_R, BtPA)
    
    # Quick stability check: trace of S = (A+BK)^T P (A+BK) - P should be negative
    Acl = A + B @ K
    S = Acl.T @ (P @ Acl) - P
    tr = 0.0
    for i in range(n):
        tr += S[i, i]
    if tr >= -1e-10:
        return False, np.empty((m, n)), np.empty((n, n))
    
    return True, K, P

# Trigger compilation at import time
_dA = np.eye(2)
_dB = np.eye(2)
_solve_all(_dA, _dB, 2, 2)
_dA1 = np.eye(1)
_dB1 = np.eye(1)
_solve_all(_dA1, _dB1, 1, 1)

class Solver:
    def solve(self, problem, **kwargs):
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        n = A.shape[0]
        m = B.shape[1]
        
        try:
            success, K, P = _solve_all(A, B, n, m)
            
            if success:
                return {
                    "is_stabilizable": True,
                    "K": K.tolist(),
                    "P": P.tolist()
                }
            else:
                return {"is_stabilizable": False, "K": None, "P": None}
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}