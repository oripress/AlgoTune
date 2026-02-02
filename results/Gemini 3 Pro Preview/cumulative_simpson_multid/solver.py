import numpy as np
from scipy.integrate import cumulative_simpson

class Solver:
    def solve(self, problem, **kwargs):
        y2 = problem["y2"]
        dx = problem["dx"]
        result = cumulative_simpson(y2, dx=dx)
        return result