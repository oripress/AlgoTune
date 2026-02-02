from __future__ import annotations

from binascii import b2a_base64 as _b2a_base64
from typing import Any

import numpy as np

try:
    import os

    import numba
    from numba import njit, prange  # type: ignore
except Exception:  # pragma: no cover
    numba = None  # type: ignore
    njit = None  # type: ignore
    prange = range  # type: ignore

# Base64 alphabet as uint8 for fast indexing in numba.
_B64_LUT = np.frombuffer(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/", dtype=np.uint8
)

if njit is not None:

    @njit(cache=True, nogil=True)
    def _b64_encode_serial(inp: np.ndarray) -> np.ndarray:
        lut = _B64_LUT
        n = inp.size
        out_len = ((n + 2) // 3) * 4
        out = np.empty(out_len, dtype=np.uint8)

        i = 0
        o = 0
        while i + 3 <= n:
            b0 = np.uint32(inp[i])
            b1 = np.uint32(inp[i + 1])
            b2 = np.uint32(inp[i + 2])
            v = (b0 << 16) | (b1 << 8) | b2
            out[o] = lut[(v >> 18) & 63]
            out[o + 1] = lut[(v >> 12) & 63]
            out[o + 2] = lut[(v >> 6) & 63]
            out[o + 3] = lut[v & 63]
            i += 3
            o += 4

        rem = n - i
        if rem == 1:
            v = np.uint32(inp[i]) << 16
            out[o] = lut[(v >> 18) & 63]
            out[o + 1] = lut[(v >> 12) & 63]
            out[o + 2] = 61  # ord('=')
            out[o + 3] = 61
        elif rem == 2:
            b0 = np.uint32(inp[i])
            b1 = np.uint32(inp[i + 1])
            v = (b0 << 16) | (b1 << 8)
            out[o] = lut[(v >> 18) & 63]
            out[o + 1] = lut[(v >> 12) & 63]
            out[o + 2] = lut[(v >> 6) & 63]
            out[o + 3] = 61
        return out

    @njit(cache=True, parallel=True, nogil=True)
    def _b64_encode_parallel(inp: np.ndarray) -> np.ndarray:
        lut = _B64_LUT
        n = inp.size
        full = n // 3
        out_len = ((n + 2) // 3) * 4
        out = np.empty(out_len, dtype=np.uint8)

        for g in prange(full):
            i = g * 3
            o = g * 4
            b0 = np.uint32(inp[i])
            b1 = np.uint32(inp[i + 1])
            b2 = np.uint32(inp[i + 2])
            v = (b0 << 16) | (b1 << 8) | b2
            out[o] = lut[(v >> 18) & 63]
            out[o + 1] = lut[(v >> 12) & 63]
            out[o + 2] = lut[(v >> 6) & 63]
            out[o + 3] = lut[v & 63]

        rem = n - full * 3
        if rem:
            o = full * 4
            i = full * 3
            if rem == 1:
                v = np.uint32(inp[i]) << 16
                out[o] = lut[(v >> 18) & 63]
                out[o + 1] = lut[(v >> 12) & 63]
                out[o + 2] = 61
                out[o + 3] = 61
            else:  # rem == 2
                b0 = np.uint32(inp[i])
                b1 = np.uint32(inp[i + 1])
                v = (b0 << 16) | (b1 << 8)
                out[o] = lut[(v >> 18) & 63]
                out[o + 1] = lut[(v >> 12) & 63]
                out[o + 2] = lut[(v >> 6) & 63]
                out[o + 3] = 61

        return out

class Solver:
    __slots__ = ("_parallel_thr",)

    def __init__(self) -> None:
        # Use C binascii for small/medium payloads; Numba parallel only for large.
        self._parallel_thr = 1 << 17  # 131072

        # Trigger compilation and set max threads during init (excluded from solve runtime).
        if numba is not None:
            try:
                numba.set_num_threads(os.cpu_count() or 1)
            except Exception:
                pass

        if njit is not None:
            z = np.empty(0, dtype=np.uint8)
            _b64_encode_serial(z)
            _b64_encode_parallel(z)

    def solve(self, problem: Any, _b2a=_b2a_base64, **kwargs) -> Any:
        pt: bytes = problem["plaintext"]
        if njit is not None and len(pt) >= self._parallel_thr:
            arr = np.frombuffer(pt, dtype=np.uint8)
            return {"encoded_data": _b64_encode_parallel(arr).tobytes()}
        return {"encoded_data": _b2a(pt, newline=False)}