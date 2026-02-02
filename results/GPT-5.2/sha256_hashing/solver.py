from __future__ import annotations

from typing import Any

import hashlib

# Copying a pre-initialized context can be faster than constructing a new one
# (especially for many small messages).
_BASE = hashlib.sha256()

class Solver:
    __slots__ = ("_out",)

    def __init__(self) -> None:
        # Reuse output dict to avoid per-call allocation overhead.
        self._out: dict[str, bytes] = {"digest": b""}

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> dict[str, bytes]:
        h = _BASE.copy()
        h.update(problem["plaintext"])
        self._out["digest"] = h.digest()
        return self._out