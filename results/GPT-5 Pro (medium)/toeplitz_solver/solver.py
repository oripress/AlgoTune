from typing import Any, Dict, List

import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: Dict[str, List[float]], **kwargs) -> List[float]:
        """
        Solve the linear system Tx = b for a Toeplitz matrix T using an optimized approach.

        - Uses scipy.linalg.solve_toeplitz (Levinson-Durbin) for O(n^2) performance.
        - Disables check_finite for speed (inputs are assumed clean by the task).
        - Uses Hermitian shortcut when r == conj(c), which is faster.

        :param problem: A dict with keys "c", "r", "b" representing the Toeplitz system.
        :return: The solution vector x as a list of floats.
        """
        c = np.asarray(problem["c"], dtype=np.float64)
        r = np.asarray(problem["r"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)

        # Fast path for Hermitian/symmetric Toeplitz: r == conj(c)
        # Use exact equality to avoid incorrect assumptions due to numerical noise.
        if c.shape == r.shape and np.array_equal(r, np.conjugate(c)):
            x = solve_toeplitz(c, b, check_finite=False)
        else:
            x = solve_toeplitz((c, r), b, check_finite=False)

        return x.tolist()