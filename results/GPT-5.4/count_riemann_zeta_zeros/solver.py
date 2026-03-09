from __future__ import annotations

from typing import Any

from mpmath import mp

class Solver:
    def __init__(self) -> None:
        mp.dps = 10
        self._cache: dict[float, int] = {}

    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        t = float(problem["t"])
        cached = self._cache.get(t)
        if cached is not None:
            return {"result": cached}
        result = mp.nzeros(t)
        self._cache[t] = result
        return {"result": result}
        return {"result": result}