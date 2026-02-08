import numpy as np
from scipy.linalg import eigh

class Solver:
    def solve(self, problem, **kwargs):
        A = np.array(problem["A"], dtype=np.float64)
        n = A.shape[0]
        
        if n == 0:
            return {"X": A}
        
        if n == 1:
            return {"X": np.maximum(A, 0.0)}
        
        # Use eigh with evd driver (divide-and-conquer, fastest for all eigenvalues)
        # check_finite=False skips input validation
        # overwrite_a=True allows modification of input for speed
        eigvals, eigvecs = eigh(A, overwrite_a=True, check_finite=False, driver='evd')
        
        # Find first positive eigenvalue (eigenvalues are sorted ascending)
        first_pos = 0
        for i in range(n):
            if eigvals[i] > 0:
                first_pos = i
                break
        else:
            # All eigenvalues are non-positive
            return {"X": np.zeros((n, n), dtype=np.float64)}
        
        if first_pos == 0:
            # All eigenvalues are non-negative, already PSD
            # But some might be zero - still need to zero them out
            # Actually if all >= 0, the matrix is already PSD
            return {"X": A}
        
        # Only use positive eigenvalues and their eigenvectors
        pos_vecs = eigvecs[:, first_pos:]
        pos_vals = eigvals[first_pos:]
        
        # X = V * diag(lambda) * V^T
        scaled = pos_vecs * pos_vals
        X = scaled @ pos_vecs.T
        
        return {"X": X}