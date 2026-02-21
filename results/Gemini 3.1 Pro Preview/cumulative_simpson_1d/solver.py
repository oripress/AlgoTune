import numpy as np
from solver_cython import compute_cumulative_simpson

class Solver:
    def solve(self, problem: dict):
        y = np.array(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        return compute_cumulative_simpson(y, dx)