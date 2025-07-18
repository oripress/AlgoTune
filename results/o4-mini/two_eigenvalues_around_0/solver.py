import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import eigsh, LinearOperator, ArpackNoConvergence

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solve by finding the two eigenvalues closest to zero.
        Uses shift-invert ARPACK for larger matrices and fast float32 EVD for others.
        """
        mx = problem["matrix"]
        N = len(mx)
        # Shift-invert ARPACK branch for medium/large N
        if N > 150:
            # Convert once to float64
            A = np.array(mx, dtype=np.float64)
            try:
                # LU factorization for fast solves
                lufac = lu_factor(A)
                def mv(x):
                    return lu_solve(lufac, x)
                OP_inv = LinearOperator((N, N), matvec=mv, dtype=np.float64)
                # ARPACK: largest magnitude of inv(A) gives eigenvalues nearest zero
                mu = eigsh(
                    OP_inv, k=2, which='LM',
                    tol=1e-4, maxiter=N,
                    ncv=2 * 2 + 1,
                    return_eigenvectors=False
                )
                lam = 1.0 / mu
                idx = np.argsort(np.abs(lam))
                return [float(lam[idx[0]]), float(lam[idx[1]])]
            except (ArpackNoConvergence, Exception):
                # fallback on failure
                pass
        # Fallback: full symmetric eigendecomposition in float32 for speed
        A32 = np.asarray(mx, dtype=np.float32)
        vals32 = np.linalg.eigvalsh(A32)
        # find two closest to zero
        idx2 = np.argpartition(np.abs(vals32), 2)[:2]
        two = vals32[idx2]
        order = np.argsort(np.abs(two))
        return [float(two[order[0]]), float(two[order[1]])]