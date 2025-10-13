from __future__ import annotations

from typing import Any
import numpy as np

try:
    # Newer, fast pocketfft with multithreading support
    from scipy import fft as sfft  # type: ignore
    _HAS_SFFT = True
except Exception:
    sfft = None  # type: ignore
    _HAS_SFFT = False

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the N-dimensional FFT quickly.

        Strategy:
        - Prefer scipy.fft (multithreaded pocketfft) when available.
        - Use workers=-1 automatically for sufficiently large arrays.
        - Fall back to NumPy's fft as needed.
        """
        a = np.asarray(problem)
        # Ensure contiguous memory for best FFT performance
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)

        # Size threshold for enabling multithreading
        size = a.size
        auto_workers = -1 if size >= 4096 else None

        # Extract worker count if provided
        user_workers = kwargs.pop("workers", None)
        workers = user_workers if user_workers is not None else auto_workers

        # Allowed keyword filters
        def filt(d: dict, allowed: tuple) -> dict:
            return {k: v for k, v in d.items() if k in allowed and v is not None}

        if _HAS_SFFT:
            try:
                if a.ndim == 1:
                    kw = filt(kwargs, ("n", "axis", "norm"))
                    if workers is not None:
                        return sfft.fft(a, workers=workers, **kw)
                    return sfft.fft(a, **kw)
                elif a.ndim == 2:
                    kw = filt(kwargs, ("s", "axes", "norm"))
                    if workers is not None:
                        return sfft.fft2(a, workers=workers, **kw)
                    return sfft.fft2(a, **kw)
                else:
                    kw = filt(kwargs, ("s", "axes", "norm"))
                    if workers is not None:
                        return sfft.fftn(a, workers=workers, **kw)
                    return sfft.fftn(a, **kw)
            except TypeError:
                # Older SciPy without 'workers' support or mismatched kwargs
                if a.ndim == 1:
                    return sfft.fft(a)
                elif a.ndim == 2:
                    return sfft.fft2(a)
                else:
                    return sfft.fftn(a)

        # Fallback to NumPy's fft (pocketfft)
        if a.ndim == 1:
            return np.fft.fft(a, **filt(kwargs, ("n", "axis", "norm")))
        if a.ndim == 2:
            # NumPy doesn't accept 'workers'; ignore if present
            return np.fft.fft2(a, **filt(kwargs, ("s", "axes", "norm")))
        return np.fft.fftn(a, **filt(kwargs, ("s", "axes", "norm")))