import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def __init__(self):
        # Pre-import and warm up
        pass
    
    def solve(self, problem, **kwargs):
        """Solve the linear system Tx = b."""
        c = np.asarray(problem["c"], dtype=np.float64)
        r = np.asarray(problem["r"], dtype=np.float64)
        b = np.asarray(problem["b"], dtype=np.float64)
        
        x = solve_toeplitz((c, r), b, check_finite=False)
        return x.tolist()