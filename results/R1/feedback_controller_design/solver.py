import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def dare_doubling_optimized(A, B, max_iter=20, tol=1e-6):
    """Highly optimized doubling algorithm with reduced operations"""
    n = A.shape[0]
    I_n = np.eye(n)
    BBT = B @ B.T
    
    # Initialize matrices
    A_k = A.copy()
    G_k = BBT.copy()
    P_k = I_n.copy()
    
    for _ in range(max_iter):
        # Precompute common terms
        T = I_n + G_k @ P_k
        try:
            # Efficient inversion using Cholesky
            L_chol = np.linalg.cholesky(T)
            T_inv = np.linalg.inv(L_chol)
            T_inv = T_inv.T @ T_inv
        except:
            # Use LU if Cholesky fails
            T_inv = np.linalg.inv(T)
        
        # Update matrices
        A_next = A_k @ T_inv @ A_k
        G_next = G_k + A_k @ T_inv @ G_k @ A_k.T
        P_next = P_k + A_k.T @ P_k @ T_inv @ A_k
        
        # Check convergence using max norm
        max_diff = np.max(np.abs(P_next - P_k))
        if max_diff < tol:
            return P_next
            
        # Update for next iteration
        A_k = A_next
        G_k = G_next
        P_k = P_next
        
    return P_k

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        B = np.array(problem["B"], dtype=np.float64)
        n = A.shape[0]
        m = B.shape[1]
        
        try:
            # Use optimized doubling algorithm
            P = dare_doubling_optimized(A, B)
            
            # Efficient K calculation
            # M = I + BᵀPB
            M = np.eye(m) + B.T @ P @ B
            # N = BᵀPA
            N = B.T @ P @ A
            
            # Solve K = -M⁻¹N
            K = -np.linalg.solve(M, N)
            
            return {
                "is_stabilizable": True,
                "K": K.tolist(),
                "P": P.tolist()
            }
        except np.linalg.LinAlgError:
            return {"is_stabilizable": False, "K": None, "P": None}