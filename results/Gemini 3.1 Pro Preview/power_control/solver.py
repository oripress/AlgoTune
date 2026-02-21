from typing import Any
from cython_solver import solve_cython

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        P, success, obj = solve_cython(
            problem["G"], 
            problem["σ"], 
            problem["P_min"], 
            problem["P_max"], 
            float(problem["S_min"])
        )
        
        if not success:
            raise ValueError("Solver failed: constraints violated")

        return {"P": P, "objective": obj}