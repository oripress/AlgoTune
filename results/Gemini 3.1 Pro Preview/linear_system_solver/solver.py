from typing import Any
from solver_cy import solve_cy

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[float]:
        return solve_cy(problem["A"], problem["b"])