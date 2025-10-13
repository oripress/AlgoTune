from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    # scipy.fft is typically faster than numpy.fft and supports multithreading.
    from scipy import fft as sp_fft  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    sp_fft = None

class Solver:
    def solve(self, problem: ArrayLike, **kwargs) -> NDArray:
        """
        Compute the N-dimensional FFT for a real-valued matrix.

        Prefer scipy.fft with multithreading; fallback to NumPy.
        """
        x = np.ascontiguousarray(problem)

        # Use scipy.fft if available (usually faster), with multithreading.
        if sp_fft is not None:
            if x.ndim == 2:
                try:
                    return sp_fft.fft2(x, workers=-1)
                except TypeError:
                    return sp_fft.fft2(x)
            else:
                try:
                    return sp_fft.fftn(x, workers=-1)
                except TypeError:
                    return sp_fft.fftn(x)

        # Fallback to NumPy's FFT (also PocketFFT-backed). Use multithreading if supported.
        if x.ndim == 2:
            try:
                return np.fft.fft2(x, workers=-1)  # type: ignore[call-arg]
            except TypeError:
                return np.fft.fft2(x)
        else:
            try:
                return np.fft.fftn(x, workers=-1)  # type: ignore[call-arg]
            except TypeError:
                return np.fft.fftn(x)