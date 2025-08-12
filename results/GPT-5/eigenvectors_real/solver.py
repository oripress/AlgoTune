from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from scipy.linalg import eigh as sp_eigh  # type: ignore
except Exception:  # pragma: no cover
    sp_eigh = None  # type: ignore

class Solver:
    def solve(self, problem, **kwargs) -> Tuple[List[float], List[List[float]]]:
        """
        Solve the eigenvalue problem for a real symmetric matrix.

        Returns:
            (eigenvalues, eigenvectors)
            - eigenvalues: list of floats in descending order
            - eigenvectors: list of lists, where each inner list is the corresponding
              normalized eigenvector (rows), forming an orthonormal set.
        """
        # Determine size without forcing copies
        if isinstance(problem, np.ndarray):
            n = problem.shape[0]
        else:
            n = len(problem)

        # Small-size fast paths without LAPACK overhead
        if n == 1:
            a00 = float(problem[0][0] if not isinstance(problem, np.ndarray) else problem[0, 0])
            return [a00], [[1.0]]

        if n == 2:
            if isinstance(problem, np.ndarray):
                a = float(problem[0, 0])
                d = float(problem[1, 1])
                b = float(0.5 * (problem[0, 1] + problem[1, 0]))
            else:
                a = float(problem[0][0])
                d = float(problem[1][1])
                b = float(0.5 * (problem[0][1] + problem[1][0]))

            tr = a + d
            diff = a - d
            s = (diff * diff + 4.0 * b * b) ** 0.5

            lam1 = 0.5 * (tr + s)
            lam2 = 0.5 * (tr - s)

            v1x = b
            v1y = lam1 - a
            if abs(v1x) < 1e-18 and abs(v1y) < 1e-18:
                if a >= d:
                    v1x, v1y = 1.0, 0.0
                else:
                    v1x, v1y = 0.0, 1.0
            norm1 = (v1x * v1x + v1y * v1y) ** 0.5
            v1x /= norm1
            v1y /= norm1

            v2x = -v1y
            v2y = v1x

            return [lam1, lam2], [[v1x, v1y], [v2x, v2y]]

        # General case: prefer SciPy's eigh with check_finite=False if available
        if sp_eigh is not None and n >= 3:
            A = np.array(problem, dtype=np.float64, order="F", copy=False)
            try:
                w, V = sp_eigh(A, lower=True, check_finite=False, overwrite_a=False, driver="evd")
            except TypeError:
                try:
                    w, V = sp_eigh(A, lower=True, check_finite=False, overwrite_a=False, turbo=True)
                except TypeError:
                    w, V = sp_eigh(A, lower=True, check_finite=False, overwrite_a=False)
        else:
            A = np.asarray(problem)
            w, V = np.linalg.eigh(A)

        # Descending order with reversed views
        eigenvalues_list = w[::-1].tolist()
        eigenvectors_list = (V.T)[::-1].tolist()

        return (eigenvalues_list, eigenvectors_list)