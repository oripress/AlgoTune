from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import scipy.linalg as la

class Solver:
    """
    Generalized eigen solver optimized vs reference:
      - uses scipy.linalg.eig(A, B) with check_finite=False and overwrite flags
      - vectorized eigenvector normalization
      - vectorized sorting (real desc, then imag desc)
    """

    def __init__(self) -> None:
        self._asarray = np.asarray
        self._norm = np.linalg.norm
        self._lexsort = np.lexsort
        self._eig = la.eig

    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs: Any) -> Any:
        A, B = problem

        A = self._asarray(A, dtype=float)
        B = self._asarray(B, dtype=float)

        # Same scaling as reference (cheap vs eig); helps numerical stability.
        scale = float(np.sqrt(self._norm(B)))
        if not np.isfinite(scale) or scale == 0.0:
            scale = 1.0
        inv_scale = 1.0 / scale

        # Create scaled copies that SciPy may overwrite.
        A_s = A * inv_scale
        B_s = B * inv_scale

        # Generalized eig (QZ); avoid finite checks for speed.
        w, v = self._eig(A_s, B_s, check_finite=False, overwrite_a=True, overwrite_b=True)

        # Normalize eigenvectors (columns) to unit norm.
        norms = self._norm(v, axis=0)
        norms = np.where(norms > 1e-300, norms, 1.0)
        v = v / norms

        # Sort by descending real part, then descending imaginary part.
        idx = self._lexsort((-w.imag, -w.real))
        w = w[idx]
        v_out = v[:, idx].T  # output expects list of eigenvectors

        return (w.tolist(), v_out.tolist())