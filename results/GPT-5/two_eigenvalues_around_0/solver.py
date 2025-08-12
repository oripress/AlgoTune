from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Find the two eigenvalues of a symmetric matrix that are closest to zero.
        Returns them sorted by absolute value (ascending).
        """
        # Use Fortran order to align with LAPACK; copy=False keeps efficiency for ndarray inputs
        A = np.array(problem["matrix"], dtype=np.float64, order="F", copy=False)
        n = A.shape[0]

        # Fast analytic path for 2x2 symmetric matrices
        if n == 2:
            a = float(A[0, 0])
            b = float(A[0, 1])
            d = float(A[1, 1])
            t = 0.5 * (a + d)
            q = 0.5 * np.hypot(a - d, 2.0 * b)  # robust sqrt((a-d)^2 + 4*b^2)
            v0 = t - q
            v1 = t + q
            if abs(v0) <= abs(v1):
                return [v0, v1]
            else:
                return [v1, v0]

        # General case: compute all eigenvalues (sorted ascending)
        ev = np.linalg.eigvalsh(A, UPLO="L")

        # Choose two nearest to zero without sorting by absolute value
        i = int(np.searchsorted(ev, 0.0, side="left"))

        if i <= 0:
            # All non-negative
            return [float(ev[0]), float(ev[1])]
        if i >= n:
            # All non-positive: last two are closest to zero
            v1 = float(ev[n - 1])
            v2 = float(ev[n - 2])
            if abs(v1) <= abs(v2):
                return [v1, v2]
            else:
                return [v2, v1]

        # Mixed signs around zero
        left = ev[i - 1]
        right = ev[i]
        if abs(left) <= abs(right):
            if i - 2 >= 0:
                left2 = ev[i - 2]
                if abs(right) <= abs(left2):
                    return [float(left), float(right)]
                else:
                    return [float(left2), float(left)]
            else:
                return [float(left), float(right)]
        else:
            if i + 1 < n:
                right2 = ev[i + 1]
                if abs(left) <= abs(right2):
                    return [float(right), float(left)]
                else:
                    return [float(right), float(right2)]
            else:
                return [float(right), float(left)]