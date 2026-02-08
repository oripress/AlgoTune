import numpy as np
import scipy.linalg
from scipy.linalg import schur
from numba import njit

@njit(cache=True)
def _tri_sqrtm(T, n):
    """Compute square root of upper triangular matrix T."""
    R = np.zeros((n, n), dtype=np.complex128)
    
    # Diagonal elements
    for i in range(n):
        R[i, i] = np.sqrt(T[i, i])
    
    # Off-diagonal elements, column by column
    for j in range(1, n):
        for i in range(j - 1, -1, -1):
            s = T[i, j]
            for k in range(i + 1, j):
                s -= R[i, k] * R[k, j]
            denom = R[i, i] + R[j, j]
            if denom == 0:
                R[i, j] = 0.0
            else:
                R[i, j] = s / denom
    
    return R

# Warm up numba
_dummy = np.eye(3, dtype=np.complex128)
_tri_sqrtm(_dummy, 3)

def _blocked_tri_sqrtm(T, n, block_size=64):
    """Compute square root of upper triangular matrix using blocked Sylvester approach."""
    R = np.zeros((n, n), dtype=np.complex128)
    
    # Number of blocks
    nblocks = (n + block_size - 1) // block_size
    blocks = []
    for i in range(nblocks):
        start = i * block_size
        end = min((i + 1) * block_size, n)
        blocks.append((start, end))
    
    # Compute diagonal blocks
    for bi, (start, end) in enumerate(blocks):
        bsize = end - start
        Tii = T[start:end, start:end]
        if bsize == 1:
            R[start, start] = np.sqrt(Tii[0, 0])
        else:
            # For small diagonal blocks, use the point-wise algorithm
            Rii = _tri_sqrtm(Tii, bsize)
            R[start:end, start:end] = Rii
    
    # Compute off-diagonal blocks using Sylvester equation
    # R_ij = solve Sylvester: R_ii @ R_ij + R_ij @ R_jj = T_ij - sum_{k=i+1}^{j-1} R_ik @ R_kj
    for col in range(1, nblocks):
        for row in range(col - 1, -1, -1):
            rs, re = blocks[row]
            cs, ce = blocks[col]
            
            # RHS = T[rs:re, cs:ce]
            C = T[rs:re, cs:ce].copy()
            
            # Subtract contributions from intermediate blocks
            for k in range(row + 1, col):
                ks, ke = blocks[k]
                C -= R[rs:re, ks:ke] @ R[ks:ke, cs:ce]
            
            # Solve Sylvester equation: R_ii @ X + X @ R_jj = C
            R_ii = R[rs:re, rs:re]
            R_jj = R[cs:ce, cs:ce]
            
            try:
                X = scipy.linalg.solve_sylvester(R_ii, R_jj, C)
                R[rs:re, cs:ce] = X
            except Exception:
                # Fallback: use the point-wise method for this block
                for j in range(cs, ce):
                    for i in range(min(re, j) - 1, rs - 1, -1):
                        s = T[i, j]
                        for kk in range(i + 1, j):
                            s -= R[i, kk] * R[kk, j]
                        denom = R[i, i] + R[j, j]
                        if denom == 0:
                            R[i, j] = 0.0
                        else:
                            R[i, j] = s / denom
    
    return R

class Solver:
    def solve(self, problem, **kwargs):
        A = problem["matrix"]
        if not isinstance(A, np.ndarray):
            A = np.array(A, dtype=np.complex128)
        elif A.dtype != np.complex128:
            A = A.astype(np.complex128)
        
        n = A.shape[0]
        
        if n == 0:
            return {"sqrtm": {"X": []}}
        
        if n == 1:
            val = np.sqrt(A[0, 0])
            return {"sqrtm": {"X": [[val]]}}
        
        # Complex Schur decomposition: A = Q T Q^H
        T, Q = schur(A, output='complex')
        
        # Choose algorithm based on matrix size
        if n <= 128:
            R = _tri_sqrtm(T, n)
        else:
            R = _blocked_tri_sqrtm(T, n, block_size=64)
        
        # Transform back: X = Q R Q^H
        X = (Q @ R) @ Q.conj().T
        
        return {"sqrtm": {"X": X.tolist()}}