from __future__ import annotations

from typing import Any

import numpy as np

try:
    # scipy.fft is typically faster and supports multithreading
    import scipy.fft as sp_fft  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback if scipy not available
    sp_fft = None

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the N-dimensional FFT of a real-valued matrix efficiently.

        - Accepts list-like or ndarray inputs.
        - Prefers SciPy's pocketfft backend (scipy.fft) with multithreading.
        - Falls back to NumPy's pocketfft (numpy.fft) if SciPy is unavailable.
        - Supports kwargs: s, axes, norm; attempts to use 'workers' when available.
        """
        # Convert input to ndarray with efficient dtypes
        a = np.asarray(problem)
        # Promote to efficient FFT dtypes
        if np.iscomplexobj(a):
            if a.dtype != np.complex128:
                a = a.astype(np.complex128, copy=False)
        else:
            if a.dtype != np.float64:
                a = a.astype(np.float64, copy=False)

        # Ensure contiguous memory for better FFT performance
        if not a.flags.c_contiguous:
            a = np.ascontiguousarray(a)

        # Extract parameters
        s = kwargs.get("s", None)
        axes = kwargs.get("axes", None)
        norm = kwargs.get("norm", None)

        call_kwargs = {}
        if s is not None:
            call_kwargs["s"] = s
        if axes is not None:
            call_kwargs["axes"] = axes
        if norm is not None:
            call_kwargs["norm"] = norm

        # Heuristic to decide whether to use threaded FFT
        def _should_multithread(arr: np.ndarray) -> bool:
            # Avoid threading overhead on tiny transforms
            if arr.size < 4096:
                return False
            # Prefer threading when at least one axis is reasonably large
            if arr.ndim >= 1 and max(arr.shape) >= 64:
                return True
            return False

        workers = kwargs.get("workers", None)
        if workers is None and _should_multithread(a):
            # Use all cores when beneficial
            try:
                import os

                cpu = os.cpu_count() or 1
            except Exception:
                cpu = 1
            if cpu > 1:
                call_kwargs["workers"] = cpu

        # Prefer SciPy's FFT if available
        if sp_fft is not None:
            try:
                if a.ndim == 2 and s is None and axes is None:
                    return sp_fft.fft2(a, **call_kwargs)
                return sp_fft.fftn(a, **call_kwargs)
            except TypeError:
                # 'workers' not supported in this version; retry without it
                call_kwargs.pop("workers", None)
                if a.ndim == 2 and s is None and axes is None:
                    return sp_fft.fft2(a, **call_kwargs)
                return sp_fft.fftn(a, **call_kwargs)

        # Fallback to NumPy FFT (also pocketfft)
        try:
            if a.ndim == 2 and s is None and axes is None:
                return np.fft.fft2(a, **call_kwargs)
            return np.fft.fftn(a, **call_kwargs)
        except TypeError:
            # Older NumPy without 'workers'
            call_kwargs.pop("workers", None)
            if a.ndim == 2 and s is None and axes is None:
                return np.fft.fft2(a, **call_kwargs)
            return np.fft.fftn(a, **call_kwargs)
        return np.fft.fftn(a, s=s, axes=axes, norm=norm)