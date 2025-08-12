import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict, **kwargs):
        """
        Solve Tx = b where T is a Toeplitz matrix defined by first column `c`
        and first row `r`. Utilizes SciPy's Levinson-Durbin implementation.
        """
        c = np.asarray(problem["c"], dtype=float)
        r = np.asarray(problem["r"], dtype=float)
        b = np.asarray(problem["b"], dtype=float)

        # SciPy's solve_toeplitz expects a tuple (c, r)
        x = solve_toeplitz((c, r), b)

        return x.tolist()