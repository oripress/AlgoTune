import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs):
        A = np.asarray(problem["A"], dtype=np.float64)
        n = A.shape[0]
        
        # For small matrices, direct numpy might be faster
        if n <= 50:
            eigvals, eigvecs = np.linalg.eigh(A)
        else:
            # Use scipy's eigh with divide-and-conquer driver and skip finite checks
            eigvals, eigvecs = eigh(A, driver='evd', check_finite=False, overwrite_a=True)
        
        # Find index where eigenvalues become positive
        # eigh returns eigenvalues in ascending order
        pos_idx = np.searchsorted(eigvals, 0.0, side='right')
        
        if pos_idx == n:
            # All eigenvalues are non-positive
            return {"X": np.zeros((n, n), dtype=np.float64)}
        
        if pos_idx == 0:
            # All eigenvalues are non-negative, use all
            eigvals = np.maximum(eigvals, 0.0)
            X = (eigvecs * eigvals) @ eigvecs.T
        else:
            # Only use positive eigenvalues
            pos_eigvals = eigvals[pos_idx:]
            pos_eigvecs = eigvecs[:, pos_idx:]
            X = (pos_eigvecs * pos_eigvals) @ pos_eigvecs.T
        
        return {"X": X}