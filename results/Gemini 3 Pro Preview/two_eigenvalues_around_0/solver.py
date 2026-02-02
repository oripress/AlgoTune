import numpy as np
from scipy.linalg.lapack import dgetrf, dgetrs
from scipy.sparse.linalg import eigsh, LinearOperator

class Solver:
    def solve(self, problem: dict[str, list[list[float]]]) -> list[float]:
        """
        Solve the problem by finding the two eigenvalues closest to zero.
        """
        # Use Fortran order for LAPACK optimization
        matrix = np.array(problem["matrix"], dtype=float, order='F')
        n = matrix.shape[0]
        
        # Threshold for switching to iterative solver
        if n < 300:
            eigenvalues = np.linalg.eigvalsh(matrix)
            eigenvalues_sorted = sorted(eigenvalues, key=abs)
            return [float(x) for x in eigenvalues_sorted[:2]]
            
        try:
            # Shift-invert strategy with sigma close to 0
            sigma = 1e-6
            
            # Modify diagonal in-place for shift
            indices = np.diag_indices(n)
            matrix[indices] -= sigma
            
            # LU Factorization using LAPACK directly
            # overwrite_a=1 allows in-place modification
            lu, piv, info = dgetrf(matrix, overwrite_a=1)
            
            if info != 0:
                raise RuntimeError(f"Factorization failed with info={info}")
            
            def matvec(v):
                # Solve (A - sigma I) x = v
                # trans=0 means solve A*x = v
                x, info = dgetrs(lu, piv, v, trans=0, overwrite_b=0)
                if info != 0:
                     raise RuntimeError(f"Solve failed with info={info}")
                return x
            
            # LinearOperator for (A - sigma I)^-1
            OP = LinearOperator((n, n), matvec=matvec, dtype=matrix.dtype)
            
            # Compute 2 eigenvalues with largest magnitude (closest to sigma for A)
            vals_inv = eigsh(OP, k=2, which='LM', return_eigenvectors=False, ncv=5, tol=1e-3)
            
            # Recover eigenvalues of A
            eigenvalues = (1.0 / vals_inv) + sigma
            
            eigenvalues_sorted = sorted(eigenvalues, key=abs)
            return [float(x) for x in eigenvalues_sorted]
            
        except Exception:
            # Fallback to robust full diagonalization
            # Reload matrix as it was modified
            matrix = np.array(problem["matrix"], dtype=float)
            eigenvalues = np.linalg.eigvalsh(matrix)
            eigenvalues_sorted = sorted(eigenvalues, key=abs)
            return [float(x) for x in eigenvalues_sorted[:2]]