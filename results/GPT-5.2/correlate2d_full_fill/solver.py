"""Fast 2D correlation (full, fill).

Must match:
    scipy.signal.correlate2d(a, b, mode="full", boundary="fill")

Using identity:
    correlate2d(a, b) == convolve2d(a, conj(flip(b)))

Key performance idea for this task's fixed shapes:
- For n=1 (30x30 with 8x8 kernel): direct convolution is cheapest.
- For n>=2: FFT-based convolution is dramatically faster than direct.

This avoids expensive heuristics and keeps overhead minimal.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.signal import convolve2d, fftconvolve

class Solver:
    __slots__ = ("_conv2d", "_fftconv")

    def __init__(self) -> None:
        self._conv2d = convolve2d
        self._fftconv = fftconvolve

    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        a, b = problem
        a = np.asarray(a)
        b = np.asarray(b)

        # Kernel for convolution equivalent to correlation:
        # flip in both axes and conjugate (conj is a no-op view for real arrays).
        k = b[::-1, ::-1].conj()

        # Shapes are (30*n,30*n) and (8*n,8*n).
        n = a.shape[0] // 30
        if n <= 1:
            return self._conv2d(a, k, mode="full", boundary="fill")
        return self._fftconv(a, k, mode="full")