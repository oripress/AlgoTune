import numpy as np
import cmath

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute eigenvectors of a real square matrix.
        Returns eigenvectors sorted by corresponding eigenvalues descending
        (by real part, then imaginary part) and normalized to unit norm.
        Output: list of eigenvectors (each a list of complex numbers).
        """
        A = np.asarray(problem, dtype=float)
        n = A.shape[0]
        # Handle trivial case
        if n == 1:
            return [[1.0+0j]]
        # Fast analytic solution for 2x2 matrices
        if n == 2:
            a, b = A[0,0], A[0,1]
            c, d = A[1,0], A[1,1]
            tr = a + d
            det = a*d - b*c
            disc = cmath.sqrt(tr*tr - 4*det)
            lam1 = (tr + disc) / 2
            lam2 = (tr - disc) / 2
            evals = [lam1, lam2]
            vecs = []
            for lam in evals:
                # Solve (A - lam I)v = 0
                if abs(b) > abs(c):
                    v0 = 1.0
                    v1 = (lam - a) / b if b != 0 else 0.0
                else:
                    v0 = c / (lam - d) if (lam - d) != 0 else 0.0
                    v1 = 1.0
                # Normalize the vector
                norm = cmath.sqrt(v0 * v0.conjugate() + v1 * v1.conjugate())
                if norm != 0:
                    v0 /= norm
                    v1 /= norm
                vecs.append([v0, v1])
            # Sort by eigenvalue criteria
            order = sorted(range(2), key=lambda i: (-evals[i].real, -evals[i].imag))
            return [vecs[i] for i in order]
        # General case: use NumPy
        evals, evecs = np.linalg.eig(A)
        # Sort by descending real part, then imaginary part
        idx = np.lexsort((-evals.imag, -evals.real))
        evecs = evecs[:, idx]
        # Normalize columns
        norms = np.linalg.norm(evecs, axis=0)
        norms[norms < 1e-12] = 1.0
        evecs /= norms
        # Return list of eigenvectors
        return evecs.T.tolist()