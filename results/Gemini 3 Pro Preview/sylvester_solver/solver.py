from typing import Any
import numpy as np
from scipy.linalg import hessenberg, schur
from numba import njit

@njit(fastmath=True, cache=True)
def solve_upper_hess_system(H, b):
    # Solves H x = b, where H is Upper Hessenberg (N x N)
    # b is length N
    # Modifies H and b in-place. b contains solution on exit.
    N = len(b)
    for i in range(N - 1):
        # Partial pivoting
        if np.abs(H[i+1, i]) > np.abs(H[i, i]):
            # Swap rows i and i+1 in H (cols i to N-1)
            # Vectorized swap
            for c in range(i, N):
                temp = H[i, c]
                H[i, c] = H[i+1, c]
                H[i+1, c] = temp
            # Swap in b
            temp_b = b[i]
            b[i] = b[i+1]
            b[i+1] = temp_b
            
        if H[i+1, i] != 0:
            mult = H[i+1, i] / H[i, i]
            # Row operation
            for c in range(i+1, N):
                H[i+1, c] -= mult * H[i, c]
            b[i+1] -= mult * b[i]
            
    # Back substitution
    for i in range(N - 1, -1, -1):
        sum_val = b[i]
        for c in range(i + 1, N):
            sum_val -= H[i, c] * b[c]
        b[i] = sum_val / H[i, i]

@njit(fastmath=True, cache=True)
def solve_hess_schur(H, S, F):
    # H: NxN Upper Hess
    # S: MxM Upper Tri
    # F: NxM
    # Solves H Y + Y S = F
    N, M = F.shape
    H_work = np.empty((N, N), dtype=H.dtype)
    bsize = 32 # Tuned block size
    
    for k_start in range(0, M, bsize):
        k_end = min(k_start + bsize, M)
        
        # 1. Solve for the diagonal block
        for k in range(k_start, k_end):
            # Update from previous columns within the block
            for j in range(k_start, k):
                s_jk = S[j, k]
                if s_jk != 0:
                    for i in range(N):
                        F[i, k] -= F[i, j] * s_jk
            
            # Prepare system
            s_kk = S[k, k]
            for r in range(N):
                for c in range(N):
                    H_work[r, c] = H[r, c]
                H_work[r, r] += s_kk
            
            solve_upper_hess_system(H_work, F[:, k])
        
        # 2. Update future blocks
        if k_end < M:
            # F[:, k_end:] -= F[:, k_start:k_end] @ S[k_start:k_end, k_end:]
            block_res = F[:, k_start:k_end] @ S[k_start:k_end, k_end:]
            for r in range(N):
                for c in range(M - k_end):
                    F[r, k_end + c] -= block_res[r, c]

@njit(fastmath=True, cache=True)
def solve_schur_hess(S, H, F):
    # S: NxN Upper Tri
    # H: MxM Upper Hess
    # F: NxM
    # Solves S Y + Y H = F
    N, M = F.shape
    H_work = np.empty((M, M), dtype=H.dtype)
    rhs_work = np.empty(M, dtype=F.dtype)
    bsize = 32 # Tuned block size
    
    n_blocks = (N + bsize - 1) // bsize
    
    for b_idx in range(n_blocks - 1, -1, -1):
        i_end = min((b_idx + 1) * bsize, N)
        i_start = b_idx * bsize
        
        # 1. Update block from rows below
        if i_end < N:
            # F[i_start:i_end, :] -= S[i_start:i_end, i_end:] @ F[i_end:, :]
            block_res = S[i_start:i_end, i_end:] @ F[i_end:, :]
            for r in range(i_end - i_start):
                for c in range(M):
                    F[i_start + r, c] -= block_res[r, c]
            
        # 2. Solve for the diagonal block
        for i in range(i_end - 1, i_start - 1, -1):
            # Update from rows below within the block
            for p in range(i + 1, i_end):
                s_ip = S[i, p]
                if s_ip != 0:
                    for j in range(M):
                        F[i, j] -= s_ip * F[p, j]
            
            # Prepare system
            s_ii = S[i, i]
            
            # Construct H_work = J (H + s_ii I)^T J
            for u in range(M):
                for v in range(M):
                    H_work[u, v] = H[M - 1 - v, M - 1 - u]
                H_work[u, u] += s_ii
                
            # Construct rhs_work = J F[i, :]^T
            for j in range(M):
                rhs_work[j] = F[i, M - 1 - j]
            
            solve_upper_hess_system(H_work, rhs_work)
            
            # Copy back
            for j in range(M):
                F[i, M - 1 - j] = rhs_work[j]
class Solver:
    def __init__(self):
        # Warmup JIT
        N, M = 2, 2
        H = np.eye(N, dtype=np.complex128)
        S = np.eye(M, dtype=np.complex128)
        F = np.zeros((N, M), dtype=np.complex128)
        solve_hess_schur(H, S, F)
        solve_schur_hess(S, H, F)
        
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        A = problem["A"]
        B = problem["B"]
        Q = problem["Q"]
        
        # Ensure complex types
        if A.dtype != np.complex128: A = A.astype(np.complex128)
        if B.dtype != np.complex128: B = B.astype(np.complex128)
        if Q.dtype != np.complex128: Q = Q.astype(np.complex128)
        
        N, M = Q.shape
        
        if N >= M:
            # A Hessenberg, B Schur
            H_A, U = hessenberg(A, calc_q=True)
            S_B, V = schur(B, output='complex')
            # F = U^H Q V
            F = U.conj().T @ Q @ V
            solve_hess_schur(H_A, S_B, F)
            # X = U F V^H
            X = U @ F @ V.conj().T
        else:
            # A Schur, B Hessenberg
            S_A, U = schur(A, output='complex')
            H_B, V = hessenberg(B, calc_q=True)
            # F = U^H Q V
            F = U.conj().T @ Q @ V
            solve_schur_hess(S_A, H_B, F)
            # X = U F V^H
            X = U @ F @ V.conj().T
            
        return {"X": X}