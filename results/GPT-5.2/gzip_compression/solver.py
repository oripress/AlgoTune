from __future__ import annotations

import zlib
from typing import Any, Dict

# Pre-create a "base" compressor and copy it per call (avoids deflateInit2 each time).
_BASE = zlib.compressobj(9, zlib.DEFLATED, 31)  # gzip wrapper, mtime=0 in header
_BASE_copy = _BASE.copy

_OUT: Dict[str, bytes] = {"compressed_data": b""}
_KEY = "compressed_data"

class Solver:
    __slots__ = ()

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, bytes]:
        c = _BASE_copy()
        comp = c.compress
        fl = c.flush
        _OUT[_KEY] = comp(problem["plaintext"]) + fl()
        return _OUT