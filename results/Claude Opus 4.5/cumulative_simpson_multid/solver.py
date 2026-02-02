import numpy as np
from scipy.integrate import cumulative_simpson

class Solver:
    def __init__(self):
        pass
    
    def solve(self, problem, **kwargs):
        y2 = np.asarray(problem["y2"])
        dx = problem["dx"]
        return cumulative_simpson(y2, dx=dx)