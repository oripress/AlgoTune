import numpy as np

# Try the pocketfft-based dstn first; fallback to fftpack
try:
    from scipy.fft import dstn as _dstn
    _supports_overwrite = True
except ImportError:
    from scipy.fftpack import dstn as _dstn
    _supports_overwrite = False

class Solver:
    def solve(self, problem, **kwargs):
        # Use float32 for faster computation, C-contiguous
        arr = np.asarray(problem, dtype=np.float32, order='C')
        # Perform DST-II over all axes
        if _supports_overwrite:
            res = _dstn(arr, type=2, overwrite_x=True)
        else:
            res = _dstn(arr, type=2)
        # Cast back to float64 for compatibility
        return res.astype(np.float64)