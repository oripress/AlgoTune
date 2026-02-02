from typing import Any
import solver_impl

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[float]:
        return solver_impl.solve_fast(problem["A"], problem["b"])