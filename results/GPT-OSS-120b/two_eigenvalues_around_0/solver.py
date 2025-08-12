import numpy as np
from typing import Any, List, Dict

# SciPy may not be available; import lazily.
try:
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _splinalg
except Exception:  # pragma: no cover
    _sp = None
    _splinalg = None

class Solver:
    def solve(self, problem: Dict[str, List[List[float]]], **kwargs) -> List[float]:
        """
        Compute the two eigenvalues of a symmetric matrix that are closest to zero.

        Parameters
        ----------
        problem : dict
            Must contain the key "matrix" with a symmetric (n+2)×(n+2) list‑of‑lists.

        Returns
        -------
        list[float]
            The two eigenvalues nearest to zero, sorted by absolute value.
        """
        # Convert input to a NumPy array; let any conversion error propagate as an exception.
        matrix = np.array(problem["matrix"], dtype=float, copy=False)

        # Compute all eigenvalues using the efficient dense routine.
        # For symmetric matrices, eigvalsh is both fast and numerically stable.
        eigenvalues = np.linalg.eigvalsh(matrix)

        # Select the two eigenvalues with smallest absolute values.
        if eigenvalues.size <= 2:
            closest = eigenvalues
        else:
            idx = np.argpartition(np.abs(eigenvalues), 1)[:2]
            closest = eigenvalues[idx]

        # Convert to plain Python floats and sort by absolute value for the validator.
        result = sorted([float(v) for v in closest], key=abs)
        return result