from typing import Any
from mpmath import mp

class Solver:
    def solve(self, problem, **kwargs):
        t = problem["t"]
        result = mp.nzeros(t)
        return {"result": int(result)}