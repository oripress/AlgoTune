from __future__ import annotations

from typing import Any, List

import numpy as np

try:  # pragma: no cover
    from scipy.signal import fftconvolve as _fftconvolve
    from scipy.signal import oaconvolve as _oaconvolve

    _HAS_SCIPY_SIGNAL = True
except Exception:  # pragma: no cover
    _fftconvolve = None
    _oaconvolve = None
    _HAS_SCIPY_SIGNAL = False

class Solver:
    def __init__(self, mode: str = "full"):
        self.mode = mode

    def solve(self, problem, **kwargs) -> Any:
        mode = kwargs.get("mode", getattr(self, "mode", "full"))
        if mode not in ("full", "valid", "same"):
            mode = "full"

        out: List[np.ndarray] = []
        append = out.append
        asarray = np.asarray
        np_correlate = np.correlate

        # Heuristics (empirical): use direct for tiny products, FFT for medium/large,
        # and overlap-add for very large products (where oaconvolve shines).
        DIRECT_NM_THRESHOLD = 120_000
        OA_NM_THRESHOLD = 25_000_000  # very large only

        has_sig = _HAS_SCIPY_SIGNAL and _fftconvolve is not None

        for a, b in problem:
            a_arr = a if isinstance(a, np.ndarray) else asarray(a)
            b_arr = b if isinstance(b, np.ndarray) else asarray(b)

            na = int(a_arr.shape[0])
            nb = int(b_arr.shape[0])

            if mode == "valid" and nb > na:
                continue

            nm = na * nb

            # Small: fastest is np.correlate (C, no FFT overhead)
            if (not has_sig) or nm <= DIRECT_NM_THRESHOLD:
                append(np_correlate(a_arr, b_arr, mode=mode))
                continue

            # Correlation via convolution with reversed second input.
            b_rev = b_arr[::-1]

            # Very large: overlap-add convolution
            if _oaconvolve is not None and nm >= OA_NM_THRESHOLD:
                append(_oaconvolve(a_arr, b_rev, mode=mode))
            else:
                append(_fftconvolve(a_arr, b_rev, mode=mode))

        return out