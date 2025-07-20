import numpy as np
# Try pocketfft's dctn for multithreaded FFT-based DCT-I
try:
    from scipy.fft import dctn
except ImportError:
    from scipy.fftpack import dctn

class Solver:
    def solve(self, problem, **kwargs):
        """
        Fast N-dimensional DCT Type I via pocketfft dctn if available.
        """
        # Convert to float64, C-contiguous, writable
        arr = np.require(problem, dtype=np.float64, requirements=['C','W'])
        # Try in-place FFT-based DCT
        try:
            return dctn(arr, type=1, axes=tuple(range(arr.ndim)), overwrite_x=True)
        except TypeError:
            # fallback when overwrite_x not supported
            return dctn(arr, type=1)