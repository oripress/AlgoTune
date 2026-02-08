import hashlib
from typing import Any

try:
    from sha256_fast import solve_sha256
    _has_fast = True
except ImportError:
    _has_fast = False

_sha256 = hashlib.sha256

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        if _has_fast:
            return solve_sha256(problem)
        return {"digest": _sha256(problem["plaintext"]).digest()}