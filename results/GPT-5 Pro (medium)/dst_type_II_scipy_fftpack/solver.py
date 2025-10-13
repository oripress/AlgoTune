from __future__ import annotations

from typing import Any

import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute 2D DST Type II efficiently.

        Strategy:
        - For 2D inputs, perform two 1D DSTs:
          first along the last axis (contiguous in C-order), then along the first axis.
          This ordering tends to be faster than a generic N-D routine.
        - Prefer scipy.fftpack (to match the reference) and fall back to scipy.fft with
          multithreading if available.
        - For non-2D inputs, fall back to dstn.
        """
        x = np.asarray(problem)
        if x.size == 0:
            return x.copy()

        # Try to use scipy.fftpack first to mirror the reference implementation closely.
        try:
            from scipy.fftpack import dst as _dst1d, dstn as _dstn  # type: ignore
            use_fftpack = True
        except Exception:  # pragma: no cover
            from scipy.fft import dst as _dst1d, dstn as _dstn  # type: ignore
            use_fftpack = False

        if x.ndim == 2:
            # Apply along contiguous axis first (axis=1 for C-order arrays).
            if use_fftpack:
                y = _dst1d(x, type=2, axis=1, overwrite_x=False)
                y = _dst1d(y, type=2, axis=0, overwrite_x=True)
            else:
                # scipy.fft supports multithreading via workers
                try:
                    y = _dst1d(x, type=2, axis=1, workers=-1)
                    y = _dst1d(y, type=2, axis=0, workers=-1, overwrite_x=True)
                except TypeError:
                    y = _dst1d(x, type=2, axis=1)
                    y = _dst1d(y, type=2, axis=0, overwrite_x=True)
            return y

        # General N-D fallback
        if use_fftpack:
            return _dstn(x, type=2)
        try:
            return _dstn(x, type=2, workers=-1)
        except TypeError:
            return _dstn(x, type=2)