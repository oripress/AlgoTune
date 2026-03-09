from __future__ import annotations

from typing import Any

import numpy as np
from scipy.sparse.linalg import eigsh

def _pick_two(eigenvalues: np.ndarray) -> list[float]:
    n = eigenvalues.shape[0]
    if n == 2:
        a = float(eigenvalues[0])
        b = float(eigenvalues[1])
        return [a, b] if abs(a) <= abs(b) else [b, a]

    right = int(np.searchsorted(eigenvalues, 0.0))
    left = right - 1

    if right <= 0:
        return [float(eigenvalues[0]), float(eigenvalues[1])]
    if right >= n:
        return [float(eigenvalues[-1]), float(eigenvalues[-2])]

    a = eigenvalues[left]
    b = eigenvalues[right]
    if abs(a) <= abs(b):
        first = float(a)
        left -= 1
    else:
        first = float(b)
        right += 1

    if left < 0:
        second = float(eigenvalues[right])
    elif right >= n:
        second = float(eigenvalues[left])
    elif abs(eigenvalues[left]) <= abs(eigenvalues[right]):
        second = float(eigenvalues[left])
    else:
        second = float(eigenvalues[right])

    return [first, second]

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        matrix = np.asarray(problem["matrix"], dtype=np.float64)
        n = matrix.shape[0]

        if n <= 32:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return _pick_two(eigenvalues)

        try:
            vals = eigsh(
                matrix,
                k=2,
                sigma=0.0,
                which="LM",
                return_eigenvectors=False,
                tol=1e-5,
                ncv=8,
                maxiter=max(n, 60),
                v0=np.ones(n, dtype=np.float64),
            )
            a = float(vals[0])
            b = float(vals[1])
            return [a, b] if abs(a) <= abs(b) else [b, a]
        except Exception:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return _pick_two(eigenvalues)