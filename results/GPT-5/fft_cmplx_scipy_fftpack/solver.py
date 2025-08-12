from __future__ import annotations

import os
from typing import Any

import numpy as np

# Prefer modern, optimized SciPy FFT backend when available.
try:
    from scipy import fft as sp_fft

    _FFT1D = sp_fft.fft
    _FFT2D = sp_fft.fft2
    _FFTN = sp_fft.fftn
except Exception:
    _FFT1D = np.fft.fft
    _FFT2D = np.fft.fft2
    _FFTN = np.fft.fftn

# Precompute worker/overwrite support detection to avoid per-call reflection.
import inspect

def _supports_param(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False

_SUPPORTS_WORKERS_1D = _supports_param(_FFT1D, "workers")
_SUPPORTS_WORKERS_2D = _supports_param(_FFT2D, "workers")
_SUPPORTS_WORKERS_ND = _supports_param(_FFTN, "workers")

_SUPPORTS_OVERWRITE_1D = _supports_param(_FFT1D, "overwrite_x")
_SUPPORTS_OVERWRITE_2D = _supports_param(_FFT2D, "overwrite_x")
_SUPPORTS_OVERWRITE_ND = _supports_param(_FFTN, "overwrite_x")

_CPU_COUNT = max(1, os.cpu_count() or 1)
_DEFAULT_THREADS = min(32, _CPU_COUNT)

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the N-dimensional FFT of the input using the fastest available backend.

        - Uses SciPy's pocketfft when available (fft/fft2/fftn).
        - Enables multi-threading via 'workers' for sufficiently large inputs.
        - Allows in-place overwrite when safe to reduce copies.
        """
        a = np.asarray(problem)
        ndim = a.ndim

        if ndim == 1:
            func = _FFT1D
            supports_workers = _SUPPORTS_WORKERS_1D
            supports_overwrite = _SUPPORTS_OVERWRITE_1D
        elif ndim == 2:
            func = _FFT2D
            supports_workers = _SUPPORTS_WORKERS_2D
            supports_overwrite = _SUPPORTS_OVERWRITE_2D
        else:
            func = _FFTN
            supports_workers = _SUPPORTS_WORKERS_ND
            supports_overwrite = _SUPPORTS_OVERWRITE_ND

        # Enable multi-threading when beneficial and not user-specified.
        if supports_workers and "workers" not in kwargs:
            # Lower threshold to leverage threading on moderate sizes.
            if a.size >= 4096:
                kwargs["workers"] = _DEFAULT_THREADS

        # Allow overwrite when we created a fresh array (i.e., input was not a NumPy array).
        if supports_overwrite and "overwrite_x" not in kwargs:
            if not isinstance(problem, np.ndarray):
                kwargs["overwrite_x"] = True

        return func(a, **kwargs)