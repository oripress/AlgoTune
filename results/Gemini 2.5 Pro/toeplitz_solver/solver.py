import numpy as np
from scipy.linalg import solve_toeplitz
from typing import Any

class Solver:
    def solve(self, problem: dict[str, list[float]], **kwargs) -> Any:
        """
        Solve the linear system Tx = b using scipy solve_Toeplitz.
        """
        c = problem["c"]
        r = problem["r"]
        b = problem["b"]
        x = solve_toeplitz((c, r), b)
        return x.tolist()