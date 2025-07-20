import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        """Solve the linear system Tx = b using scipy solve_toeplitz."""
        c = np.array(problem["c"])
        r = np.array(problem["r"])
        b = np.array(problem["b"])
        
        x = solve_toeplitz((c, r), b)
        return x.tolist()