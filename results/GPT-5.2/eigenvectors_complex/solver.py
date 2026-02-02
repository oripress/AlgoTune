from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    """
    Fast eigenvector solver.

    Must match the reference eigenvectors produced by `np.linalg.eig` (up to phase),
    sorted by eigenvalue real part (desc), then imaginary part (desc), with stable
    tie-breaking by original index.
    """

    def __init__(self) -> None:
        # Cache attribute lookups for speed
        self._eig = np.linalg.eig
        self._lexsort = np.lexsort
        self._asarray = np.asarray
        self._arange = np.arange

    def solve(self, problem, **kwargs) -> Any:
        A = self._asarray(problem)
        w, v = self._eig(A)  # v columns are eigenvectors

        n = w.shape[0]
        # Stable tie-break to match Python's stable sort behavior in the reference.
        idx = self._lexsort((self._arange(n), -w.imag, -w.real))

        # Return eigenvectors as a list of n lists (each length n), complex entries.
        return v[:, idx].T.tolist()