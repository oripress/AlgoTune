import numpy as np
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem):
        y2 = problem["y2"]
        dx = problem["dx"]
        return cumulative_simpson(y2, dx=dx, axis=-1)