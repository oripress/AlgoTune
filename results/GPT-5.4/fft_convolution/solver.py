from __future__ import annotations

from typing import Any

import numpy as np
from numpy.fft import irfft, rfft
from scipy.fft import next_fast_len

class Solver:
    __slots__ = ()

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> Any:
        x = np.asarray(problem["signal_x"], dtype=np.float64)
        y = np.asarray(problem["signal_y"], dtype=np.float64)
        mode = problem.get("mode", "full")

        n1 = x.size
        n2 = y.size
        if n1 == 0 or n2 == 0:
            return {"convolution": np.empty(0, dtype=np.float64)}

        if n1 * n2 <= 524288 or (n1 if n1 < n2 else n2) <= 64:
            return {"convolution": np.convolve(x, y, mode=mode)}

        full_len = n1 + n2 - 1
        fft_len = next_fast_len(full_len)
        full = irfft(rfft(x, fft_len) * rfft(y, fft_len), fft_len)[:full_len]

        if mode == "full":
            conv = full
        elif mode == "same":
            start = (n2 - 1) // 2
            conv = full[start : start + n1]
        else:
            small = n1 if n1 < n2 else n2
            start = small - 1
            conv = full[start : start + (n1 + n2 - 2 * small + 1)]

        return {"convolution": conv}