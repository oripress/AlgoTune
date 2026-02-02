"""Optimized solver for Convolve2D Full Fill.

Matches:
    scipy.signal.convolve2d(a, b, mode="full", boundary="fill")
"""

# pylint: disable=invalid-sequence-index

from __future__ import annotations

from typing import Any

import inspect
import os

import numpy as np
from scipy import signal
from scipy.fft import irfft2 as _irfft2
from scipy.fft import next_fast_len as _next_fast_len
from scipy.fft import rfft2 as _rfft2

_SIG_RFFT2 = inspect.signature(_rfft2).parameters
_SIG_IRFFT2 = inspect.signature(_irfft2).parameters
_SIG_NFL = inspect.signature(_next_fast_len).parameters

_HAS_WORKERS = "workers" in _SIG_RFFT2
_HAS_OVERWRITE = "overwrite_x" in _SIG_RFFT2 and "overwrite_x" in _SIG_IRFFT2
_HAS_REAL_NFL = "real" in _SIG_NFL

_CPU_COUNT = os.cpu_count() or 1
_DEFAULT_WORKERS = -1 if _CPU_COUNT > 1 else 1

# Enable workers for more medium-sized FFTs (tuned for typical pocketfft overhead).
_MIN_ELEMS_FOR_WORKERS = 64 * 64

# Cache next_fast_len results across calls (only a few sizes appear in this benchmark).
_FASTLEN_CACHE: dict[tuple[int, bool], int] = {}

def _fast_len(n: int, real: bool) -> int:
    key = (n, real)
    v = _FASTLEN_CACHE.get(key)
    if v is not None:
        return v
    if real and _HAS_REAL_NFL:
        v = _next_fast_len(n, real=True)
    else:
        v = _next_fast_len(n)
    _FASTLEN_CACHE[key] = v
    return v

class Solver:
    """Compute full 2D convolution with zero-fill boundary."""

    mode = "full"
    boundary = "fill"

    # Push FFT earlier (n>=3 for the task's (30n)x(30n) and (8n)x(8n) pattern).
    _DIRECT_MAX_MULS = 2_000_000

    def solve(self, problem: tuple, **kwargs) -> Any:
        a, b = problem

        m, n = a.shape
        p, q = b.shape
        out0 = m + p - 1
        out1 = n + q - 1

        if (m * n * p * q) <= self._DIRECT_MAX_MULS:
            return signal.convolve2d(a, b, mode="full", boundary="fill")

        f0 = _fast_len(out0, real=False)
        f1 = _fast_len(out1, real=True)

        # Avoid copies when possible.
        if a.dtype == np.float64 and a.flags.c_contiguous:
            a_arr = a
            ow_a = False
        else:
            a_arr = np.ascontiguousarray(a, dtype=np.float64)
            ow_a = True

        if b.dtype == np.float64 and b.flags.c_contiguous:
            b_arr = b
            ow_b = False
        else:
            b_arr = np.ascontiguousarray(b, dtype=np.float64)
            ow_b = True

        use_workers = _HAS_WORKERS and (f0 * f1 >= _MIN_ELEMS_FOR_WORKERS)
        w = _DEFAULT_WORKERS

        if _HAS_OVERWRITE:
            if use_workers:
                fa = _rfft2(a_arr, s=(f0, f1), workers=w, overwrite_x=ow_a)
                fa *= _rfft2(b_arr, s=(f0, f1), workers=w, overwrite_x=ow_b)
                conv = _irfft2(fa, s=(f0, f1), workers=w, overwrite_x=True)
            else:
                fa = _rfft2(a_arr, s=(f0, f1), overwrite_x=ow_a)
                fa *= _rfft2(b_arr, s=(f0, f1), overwrite_x=ow_b)
                conv = _irfft2(fa, s=(f0, f1), overwrite_x=True)
        else:
            if use_workers:
                fa = _rfft2(a_arr, s=(f0, f1), workers=w)
                fa *= _rfft2(b_arr, s=(f0, f1), workers=w)
                conv = _irfft2(fa, s=(f0, f1), workers=w)
            else:
                fa = _rfft2(a_arr, s=(f0, f1))
                fa *= _rfft2(b_arr, s=(f0, f1))
                conv = _irfft2(fa, s=(f0, f1))

        return conv[:out0, :out1]