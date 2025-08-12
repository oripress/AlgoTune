import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """Optimized SVD using matrix property detection."""
        # Get matrix
        A = np.array(problem["matrix"], dtype=np.float64)
        n, m = A.shape
        
        # Check if matrix is square and symmetric - use eigenvalue decomposition
        if n == m and np.allclose(A, A.T):
            eigvals, U = np.linalg.eigh(A)
            # Sort in descending order
            idx = np.argsort(eigvals)[::-1]
            s = np.abs(eigvals[idx])
            U = U[:, idx]
            V = U.copy()
        else:
            # Use standard SVD
            U, s, Vh = np.linalg.svd(A, full_matrices=False)
            V = Vh.T
        
        return {"U": U, "S": s, "V": V}