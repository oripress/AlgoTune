import numpy as np
from typing import Any
from scipy.fft import fft2
import os

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the 2‑D FFT of a real‑valued square matrix.

        The implementation first tries to use **pyFFTW** (if installed) because
        it provides highly‑optimized native code and explicit multithreading.
        If pyFFTW is unavailable, it falls back to SciPy's multi‑threaded
        ``fft2`` implementation. The input is cast to ``float32`` to
        reduce memory bandwidth.
        """
        # Convert input to float32 for lower memory bandwidth.
        arr = np.asarray(problem, dtype=np.float32)

        # Try pyFFTW first – it may not be installed.
        try:
            import pyfftw
            # Ensure pyFFTW uses all available CPU cores.
            threads = os.cpu_count() or 1
            # Use pyFFTW's NumPy‑compatible interface.
            import pyfftw.interfaces.numpy_fft as fft
            # The pyFFTW interface respects the ``overwrite_input`` flag.
            return fft.fft2(arr, overwrite_input=True, threads=threads)
        except Exception:
            # Fallback to SciPy's highly‑optimized FFT.
            return fft2(arr, overwrite_x=True, workers=-1)