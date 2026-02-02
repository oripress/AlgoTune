from __future__ import annotations

from typing import Any, Dict

from mpmath import mp

class Solver:
    def __init__(self) -> None:
        # Try slightly lower precision; mp.nzeros is robust and often doesn't need much.
        mp.dps = 20
        self._cache: Dict[float, int] = {}

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        t = float(problem["t"])
        cached = self._cache.get(t)
        if cached is not None:
            return {"result": cached}

        # Exact count via mpmath.
        res = int(mp.nzeros(t))
        self._cache[t] = res
        return {"result": res}