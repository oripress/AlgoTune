import numpy as np
from scipy.linalg import solve_toeplitz

class Solver:
    def solve(self, problem: dict[str, list[float]]) -> list[float]:
        c = problem["c"]
        r = problem["r"]
        b = problem["b"]
        x = solve_toeplitz((c, r), b)
        return x.tolist()