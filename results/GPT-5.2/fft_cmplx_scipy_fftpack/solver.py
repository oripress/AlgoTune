from __future__ import annotations

from typing import Any

import numpy as np

try:
    import scipy.fft as _sfft  # type: ignore

    _FFT2 = _sfft.fft2
    _FFTN = _sfft.fftn

    # Tuned threshold (elements) for enabling multithreading.
    _MT = 1 << 14  # 16384

    class Solver:
        __slots__ = ()

        def solve(
            self,
            problem,
            **kwargs,
        ) -> Any:
            return _solve_scipy(problem)

    def _solve_scipy(a, fft2=_FFT2, fftn=_FFTN, mt=_MT):
        s = a.size
        if a.ndim == 2:
            if s >= mt:
                return fft2(a, workers=-1)
            return fft2(a)
        if s >= mt:
            return fftn(a, workers=-1)
        return fftn(a)

except Exception:  # pragma: no cover
    _NP_FFTN = np.fft.fftn

    class Solver:
        __slots__ = ()

        def solve(self, problem, **kwargs) -> Any:
            return _NP_FFTN(problem)