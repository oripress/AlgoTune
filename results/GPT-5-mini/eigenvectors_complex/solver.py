import numpy as np
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Fast NumPy-only eigenvector solver:
        - Accepts a square matrix (list-of-lists or ndarray).
        - Uses numpy.linalg.eig on Fortran-contiguous arrays (column-major) for LAPACK.
        - Sorts eigenpairs by eigenvalue real part (desc), then imaginary part (desc).
        - Normalizes eigenvectors (columns) in-place and returns list-of-lists.
        """
        A = np.asarray(problem)

        # Empty input
        if A.size == 0:
            return []

        # If 1D, try to reshape to square
        if A.ndim == 1:
            total = A.size
            k = int(np.sqrt(total))
            if k * k == total:
                A = A.reshape((k, k))
            else:
                A = A.reshape((1, total))

        # Ensure 2D square matrix: try reshape by total size or pad with zeros
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            total = A.size
            k = int(np.sqrt(total))
            if k * k == total:
                A = A.reshape((k, k))
            else:
                if A.ndim == 2:
                    r, c = A.shape
                else:
                    r, c = 1, A.size
                    A = A.reshape((r, c))
                n = max(r, c)
                B = np.zeros((n, n), dtype=A.dtype)
                B[:r, :c] = A
                A = B

        n = A.shape[0]
        if n == 0:
            return []

        # Force column-major layout and appropriate dtype for LAPACK
        if np.iscomplexobj(A):
            A = np.asfortranarray(A, dtype=np.complex128)
        else:
            A = np.asfortranarray(A, dtype=np.float64)

        # Compute eigenpairs with NumPy (delegates to LAPACK)
        w, v = np.linalg.eig(A)

        # Sort indices: primary -real (descending), secondary -imag (descending)
        idx = np.lexsort((-w.imag, -w.real))

        # Reorder eigenvectors (columns) and normalize in-place
        v_sorted = v[:, idx]

        # Compute column norms efficiently and avoid division by zero
        norms = np.sqrt(np.einsum('ij,ij->j', np.conjugate(v_sorted), v_sorted).real)
        norms = np.where(norms < 1e-12, 1.0, norms)
        v_sorted /= norms

        # Return list of eigenvectors (each as list of complex numbers) in sorted order
        return v_sorted.T.tolist()