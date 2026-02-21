from typing import Any
from fast_clique import solve_fast

class Solver:
    def solve(self, problem: list[list[int]], **kwargs) -> Any:
        return solve_fast(problem)