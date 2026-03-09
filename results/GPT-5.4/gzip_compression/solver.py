from __future__ import annotations

from typing import Any
from zlib import compress

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, bytes]:
        return {"compressed_data": compress(problem["plaintext"], 8, 31)}