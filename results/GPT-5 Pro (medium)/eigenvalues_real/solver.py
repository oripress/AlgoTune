from __future__ import annotations

from typing import Any, List

import numpy as np
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray, **kwargs) -> List[float]:
        """
        Compute eigenvalues of a symmetric real matrix in descending order.

        Parameters:
            problem: A symmetric numpy array (n x n) with real entries.

        Returns:
            List[float]: Eigenvalues in descending order.
        """
        # Ensure array form without unnecessary copying
        a = np.asarray(problem)

        # Compute eigenvalues for symmetric/Hermitian matrices (ascending order)
        vals = np.linalg.eigvalsh(a)

        # Return in descending order as Python floats
        return vals[::-1].tolist()