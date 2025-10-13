from typing import Any, List

import numpy as np

try:
    # Prefer SciPy's generalized eigensolver if available
    from scipy import linalg as la  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    la = None  # type: ignore
    _HAS_SCIPY = False

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the generalized eigenvalue problem A x = lambda B x for possibly complex eigenpairs.

        Returns:
          - eigenvalues: list of complex, sorted by descending real part then imag part
          - eigenvectors: list of n vectors (each list of complex), unit-norm, matching eigenvalues order
        """
        A, B = problem
        # Create Fortran-contiguous copies to minimize internal LAPACK copies and allow overwrite
        Af = np.array(A, dtype=np.float64, order="F", copy=True)
        Bf = np.array(B, dtype=np.float64, order="F", copy=True)

        # Scale for numerical stability (same as reference approach)
        # Using Frobenius norm by default via np.linalg.norm on matrices.
        scale_B = np.sqrt(np.linalg.norm(Bf))
        if not (scale_B == 0 or not np.isfinite(scale_B)):
            inv_scale = 1.0 / scale_B
            Af *= inv_scale
            Bf *= inv_scale

        if _HAS_SCIPY:
            # Use SciPy's ggev via la.eig for robustness; disable check_finite for speed.
            # Allow overwriting Af/Bf to avoid extra copies in LAPACK calls.
            eigvals, eigvecs = la.eig(Af, Bf, check_finite=False, overwrite_a=True, overwrite_b=True)
        else:  # pragma: no cover
            # Fallback if SciPy unavailable: solve standard eigenproblem on B^{-1} A
            X = np.linalg.solve(Bf, Af)
            eigvals, eigvecs = np.linalg.eig(X)

        # Normalize eigenvectors (columns) to unit Euclidean norm (vectorized)
        norms = np.linalg.norm(eigvecs, axis=0)
        nz = norms > 1e-15
        if np.any(nz):
            eigvecs[:, nz] /= norms[nz]

        # Sort by descending real part, then descending imaginary part
        order = np.lexsort((-eigvals.imag, -eigvals.real))
        eigvals_sorted = eigvals[order]
        eigvecs_sorted = eigvecs[:, order]

        # Convert to Python lists
        eigenvalues_list: List[complex] = eigvals_sorted.tolist()
        eigenvectors_list: List[List[complex]] = eigvecs_sorted.T.tolist()

        return (eigenvalues_list, eigenvectors_list)