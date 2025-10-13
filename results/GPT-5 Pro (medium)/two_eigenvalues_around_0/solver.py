from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import eigvalsh as sl_eigvalsh

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Find the two eigenvalues of a symmetric matrix that are closest to zero.

        Args:
            problem (dict): Dictionary with key "matrix" containing a symmetric matrix.

        Returns:
            list[float]: Two eigenvalues closest to zero, sorted by absolute value.
        """
        # Use Fortran order to better match LAPACK expectations; allow overwrite for speed
        matrix = np.array(problem["matrix"], dtype=np.float64, order="F", copy=True)

        # Compute all eigenvalues of a symmetric (Hermitian) matrix efficiently.
        # SciPy's eigvalsh allows skipping NaN checks and overwriting input for speed.
        # Driver 'evd' (divide-and-conquer) is typically fastest for full spectrum.
        eigenvalues = sl_eigvalsh(
            matrix, lower=True, check_finite=False, overwrite_a=True, driver="evd"
        )

        n = eigenvalues.size
        if n <= 2:
            result = eigenvalues.tolist()
            result.sort(key=abs)
            return result

        # Eigenvalues are sorted ascending. The two closest to zero must be among:
        # {j-2, j-1, j, j+1}, where j is the first index with eigenvalues[j] >= 0.
        j = int(np.searchsorted(eigenvalues, 0.0, side="left"))
        cand_idx = []
        if j - 2 >= 0:
            cand_idx.append(j - 2)
        if j - 1 >= 0:
            cand_idx.append(j - 1)
        if j < n:
            cand_idx.append(j)
        if j + 1 < n:
            cand_idx.append(j + 1)

        # Pick two with smallest absolute values among candidates
        cands = [(abs(eigenvalues[i]), float(eigenvalues[i])) for i in cand_idx]
        cands.sort(key=lambda x: x[0])
        a = cands[0][1]
        b = cands[1][1]

        # Sort the two by absolute value for output consistency
        if abs(a) <= abs(b):
            return [a, b]
        else:
            return [b, a]