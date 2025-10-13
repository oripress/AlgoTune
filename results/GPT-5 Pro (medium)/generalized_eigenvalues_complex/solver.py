from __future__ import annotations

from typing import Any, List

import numpy as np
import scipy.linalg as la

class Solver:
    def solve(self, problem, **kwargs) -> List[complex]:
        """
        Solve the generalized eigenvalue problem A x = λ B x.

        Returns eigenvalues sorted in descending order by real part, then imaginary part.
        """
        A, B = problem

        # Ensure arrays; use float64 for LAPACK and Fortran order to minimize copies.
        A_arr = np.asarray(A, dtype=np.float64, order="F")
        B_arr = np.asarray(B, dtype=np.float64, order="F")

        # Scale matrices for numerical stability (scaling both leaves λ unchanged).
        norm_B = np.linalg.norm(B_arr)
        if norm_B > 0:
            inv_scale = 1.0 / np.sqrt(norm_B)
            # Make writable Fortran copies to allow in-place scaling and LAPACK overwrites.
            A_f = np.array(A_arr, copy=True, order="F")
            B_f = np.array(B_arr, copy=True, order="F")
            A_f *= inv_scale
            B_f *= inv_scale
        else:
            # If B is zero, skip scaling but still ensure Fortran-contiguous copies.
            A_f = np.array(A_arr, copy=True, order="F")
            B_f = np.array(B_arr, copy=True, order="F")

        # Compute generalized eigenvalues only (no eigenvectors) with minimal checks.
        # Using eig with left=False, right=False avoids computing eigenvectors (faster).
        try:
            w = la.eig(A_f, B_f, left=False, right=False, overwrite_a=True, overwrite_b=True, check_finite=False)
        except TypeError:
            # Fallback for SciPy versions without some keywords.
            w = la.eig(A_f, B_f, left=False, right=False)

        # Sort descending by real part, then by imaginary part using NumPy for speed.
        idx = np.lexsort((w.imag, w.real))  # ascending by real, then imag
        w_sorted = w[idx[::-1]]  # reverse for descending
        return w_sorted.tolist()