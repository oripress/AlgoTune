from typing import Any, Sequence, Tuple
import numpy as np

class Solver:
    def solve(self, problem: Tuple[Sequence, Sequence], **kwargs) -> Any:
        """
        Compute the outer product of two 1-D vectors and return an n x m numpy array.
        Uses a minimal, low-overhead broadcasting multiply to maximize speed.
        """
        if not isinstance(problem, (tuple, list)) or len(problem) != 2:
            raise ValueError("problem must be a tuple or list (vec1, vec2)")

        vec1, vec2 = problem
        a = np.asarray(vec1).ravel()
        b = np.asarray(vec2).ravel()

        # Fast empty-case
        if a.size == 0 or b.size == 0:
            return np.empty((a.size, b.size), dtype=np.result_type(a, b))

        # Single C-backed ufunc call via broadcasting
        return a[:, None] * b[None, :]