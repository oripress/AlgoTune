from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import fftconvolve as _fftconvolve
from scipy.signal import oaconvolve as _oaconvolve

class Solver:
    def __init__(self, mode: str = "full"):
        self.mode = mode
        # Tuned to favor fast C direct path (np.convolve) for small/medium sizes.
        self._direct_thresh = 750_000

        # Cache common callables locally (init time not counted).
        self._asarray = np.asarray
        self._np_convolve = np.convolve
        self._fftconvolve = _fftconvolve
        self._oaconvolve = _oaconvolve
        self._empty = np.empty
        self._result_type = np.result_type

    def solve(self, problem, **kwargs) -> Any:
        mode = self.mode if not kwargs else kwargs.get("mode", self.mode)
        direct_thresh = (
            self._direct_thresh
            if not kwargs
            else kwargs.get("direct_thresh", self._direct_thresh)
        )

        a_in, b_in = problem
        a = self._asarray(a_in)
        b = self._asarray(b_in)

        if a.ndim != 1:
            a = a.ravel()
        if b.ndim != 1:
            b = b.ravel()

        n = a.size
        m = b.size
        if n == 0 or m == 0:
            return self._empty(0, dtype=self._result_type(a, b))

        nm = n * m
        if nm <= direct_thresh:
            # Much lower overhead than scipy.signal.convolve for 1D.
            return self._np_convolve(a, b, mode=mode)

        # Very unbalanced lengths: overlap-add is often best.
        if (min(n, m) * 8) < max(n, m) and max(n, m) >= 2048:
            return self._oaconvolve(a, b, mode=mode)

        # Otherwise, call FFT convolution directly (skip convolve's method selection).
        return self._fftconvolve(a, b, mode=mode)