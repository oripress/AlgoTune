import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        c = np.fromiter(problem["c"], dtype=float)
        r = np.fromiter(problem["r"], dtype=float)
        b = np.fromiter(problem["b"], dtype=float)

        x = solve_toeplitz((c, r), b, check_finite=False)
        return x.tolist()