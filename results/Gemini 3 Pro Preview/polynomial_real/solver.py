import numpy as np
import scipy.linalg
class Solver:
    def solve(self, problem: list[float]) -> list[float]:
        """
        Solve the polynomial problem by finding all real roots of the polynomial.
        """
        # Manual implementation of np.roots to avoid overhead
        p = np.array(problem, dtype=float)
        
        # Strip leading zeros
        nz = np.flatnonzero(p != 0)
        if len(nz) == 0:
            return []
        p = p[nz[0]:]
        
        n = len(p) - 1
        if n == 0:
            return []
        if n == 1:
            return [-p[1]/p[0]]
            
        # Companion matrix
        # A = np.zeros((n, n))
        # A[1:, :-1] = np.eye(n - 1)
        # A[0, :] = -p[1:] / p[0]
        
        # Optimized construction
        A = np.zeros((n, n), order='F')
        # Fill diagonal -1
        # np.fill_diagonal(A[1:, :-1], 1) # This is slow?
        # Slicing is fast
        # A[1:, :-1] = np.eye(n - 1)
        # Faster way to set subdiagonal
        rng = np.arange(n - 1)
        A[rng + 1, rng] = 1
        A[0, :] = -p[1:] / p[0]
        
        # Compute eigenvalues using numpy.linalg.eigvals
        roots = np.linalg.eigvals(A)
        
        return np.sort(np.real(roots))[::-1].tolist()