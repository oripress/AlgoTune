import numpy as np
from numpy.typing import NDArray
from simpson_cy import cumulative_simpson_cy

class Solver:
    def solve(self, problem: dict) -> NDArray:
        y = problem["y"]
        dx = problem["dx"]
        return cumulative_simpson_cy(y, dx)