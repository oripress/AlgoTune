from typing import Any
from solver_cython import solve_cython

class Solver:
    def solve(self, problem: dict[str, list[float]], **kwargs) -> float:
        u, v = problem.values()
        return solve_cython(u, v)