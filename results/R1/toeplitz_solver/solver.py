import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem, **kwargs):
        # Convert inputs to numpy arrays
        c = np.array(problem['c'], dtype=np.float64)
        r = np.array(problem['r'], dtype=np.float64)
        b = np.array(problem['b'], dtype=np.float64)
        
        # Solve using scipy's optimized function
        x = solve_toeplitz((c, r), b, check_finite=False)
        return x.tolist()