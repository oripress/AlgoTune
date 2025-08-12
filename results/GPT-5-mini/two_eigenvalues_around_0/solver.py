import numpy as np
from typing import Any, List

class Solver:
    def solve(self, problem: dict, **kwargs) -> List[float]:
        """
        Find the two eigenvalues closest to zero for a symmetric matrix.

        Strategy:
        - For small matrices (n <= SMALL_N), compute the full spectrum with
          numpy.linalg.eigvalsh (fast for small n).
        - For larger matrices, try to compute only the two eigenvalues nearest
          to zero using scipy.sparse.linalg.eigsh with shift-invert (sigma=0),
          which is much faster for large matrices when only a few eigenvalues
          are needed.
        - On any failure or if scipy is unavailable, fall back to the dense
          eigenvalue routine.

        Returns a list with exactly two floats sorted by absolute value.
        """
        mat_list = problem.get("matrix")
        if mat_list is None:
            raise ValueError("Input problem must contain 'matrix' key")

        A = np.array(mat_list, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Input 'matrix' must be a square 2D array")

        n = A.shape[0]
        # For very small matrices, computing the full spectrum is cheap and
        # avoids scipy import overhead.
        SMALL_N = 150

        def _two_from_full():
            vals = np.linalg.eigvalsh(A)
            idx = np.argsort(np.abs(vals))
            return [float(vals[idx[0]]), float(vals[idx[1]])]

        if n <= 2:
            # trivial sizes: just compute full eigenvalues
            return _two_from_full()

        if n <= SMALL_N:
            return _two_from_full()

        # For larger matrices, try sparse/eigs approach
        try:
            # Import here to avoid heavy import cost for small problems
            from scipy.sparse.linalg import eigsh  # type: ignore

            k = 2
            if k >= n:
                return _two_from_full()

            # Use ARPACK shift-invert mode to find eigenvalues nearest sigma=0.
            # Request a reasonable tolerance for accuracy.
            try:
                # newer scipy: can request return_eigenvectors=False
                vals = eigsh(A, k=k, sigma=0.0, which="LM", tol=1e-8, return_eigenvectors=False)
            except TypeError:
                # fallback if the signature doesn't accept return_eigenvectors
                vals, _ = eigsh(A, k=k, sigma=0.0, which="LM", tol=1e-8)

            vals = np.asarray(vals, dtype=float)
            # Validate result
            if vals.size < 2 or not np.all(np.isfinite(vals)):
                # fallback to dense
                return _two_from_full()

            # Sort by absolute value and return two values
            idx = np.argsort(np.abs(vals))
            return [float(vals[idx[0]]), float(vals[idx[1]])]

        except Exception:
            # Any failure (import/convergence/etc) — fallback to dense eigen solver
            return _two_from_full()