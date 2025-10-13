from typing import Any, Tuple

import numpy as np

class Solver:
    def solve(self, problem: Tuple[Any, Any], **kwargs) -> np.ndarray:
        """
        Compute the outer product of two 1D vectors using NumPy broadcasting.

        Args:
            problem: A tuple (vec1, vec2), where each is array-like of shape (n,).

        Returns:
            A NumPy array of shape (n, n) representing the outer product.
        """
        vec1, vec2 = problem

        # Convert to 1D NumPy arrays (ravel handles cases where they are not 1D)
        v1 = np.asarray(vec1).ravel()
        v2 = np.asarray(vec2).ravel()

        # Use broadcasting for fast outer product
        return v1[:, None] * v2[None, :]