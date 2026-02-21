from typing import Any
from mpmath import mp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        t = problem["t"]
        print(f"t = {t}")
        result = mp.nzeros(t)
        return {"result": result}