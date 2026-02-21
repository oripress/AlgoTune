from typing import Any
from fast_solver import solve_cython

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        return solve_cython(problem["sample1"], problem["sample2"])