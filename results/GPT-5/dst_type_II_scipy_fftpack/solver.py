from typing import Any

import numpy as np

# Try to import the faster pocketfft backend
try:
    from scipy import fft as _sp_fft  # type: ignore
    _HAS_SCIPY_FFT_DST = hasattr(_sp_fft, "dst")
    _HAS_SCIPY_FFT_DSTN = hasattr(_sp_fft, "dstn")
except Exception:
    _sp_fft = None  # type: ignore
    _HAS_SCIPY_FFT_DST = False
    _HAS_SCIPY_FFT_DSTN = False

# Fallback to legacy fftpack
try:
    from scipy import fftpack as _sp_fftpack  # type: ignore
    _HAS_FFTPACK_DST = hasattr(_sp_fftpack, "dst")
    _HAS_FFTPACK_DSTN = hasattr(_sp_fftpack, "dstn")
except Exception:
    _sp_fftpack = None  # type: ignore
    _HAS_FFTPACK_DST = False
    _HAS_FFTPACK_DSTN = False
def _build_dstn_impl():
    """
    Build the fastest available dstn(Type-II) callable once to avoid per-call branching.
    Returns a function f(a: ndarray) -> ndarray or None if not available.
    """
    z = np.zeros((2, 2), dtype=np.float64)
    # Prefer scipy.fft (pocketfft)
    if _sp_fft is not None and _HAS_SCIPY_FFT_DSTN:
        f = _sp_fft.dstn  # type: ignore[attr-defined]
        # Try workers + overwrite_x
        try:
            _ = f(z, type=2, norm=None, workers=-1, overwrite_x=True)  # type: ignore[arg-type]
            return lambda a, f=f: f(a, type=2, norm=None, workers=-1, overwrite_x=True)  # type: ignore[arg-type]
        except Exception:
            pass
        # Try workers only
        try:
            _ = f(z, type=2, norm=None, workers=-1)  # type: ignore[arg-type]
            return lambda a, f=f: f(a, type=2, norm=None, workers=-1)  # type: ignore[arg-type]
        except Exception:
            pass
        # Try overwrite_x only
        try:
            _ = f(z, type=2, norm=None, overwrite_x=True)  # type: ignore[arg-type]
            return lambda a, f=f: f(a, type=2, norm=None, overwrite_x=True)  # type: ignore[arg-type]
        except Exception:
            pass
        # Try norm only
        try:
            _ = f(z, type=2, norm=None)  # type: ignore[arg-type]
            return lambda a, f=f: f(a, type=2, norm=None)  # type: ignore[arg-type]
        except Exception:
            pass
        # Last try without norm
        try:
            _ = f(z, type=2)  # type: ignore[arg-type]
            return lambda a, f=f: f(a, type=2)  # type: ignore[arg-type]
        except Exception:
            pass
    # Fallback to scipy.fftpack
    if _sp_fftpack is not None and _HAS_FFTPACK_DSTN:
        f = _sp_fftpack.dstn  # type: ignore[attr-defined]
        try:
            _ = f(z, type=2)  # type: ignore[arg-type]
            return lambda a, f=f: f(a, type=2)  # type: ignore[arg-type]
        except Exception:
            pass
    return None

_DSTN_IMPL = _build_dstn_impl()

def _dst2_fast(a: np.ndarray) -> np.ndarray:
    """
    Fast 2D DST-II computed as two successive 1D DSTs along the last axis
    (contiguous in C layout) and then along the second-last axis.
    """
    # Prefer scipy.fft.dst (pocketfft, multithreaded)
    if _sp_fft is not None and _HAS_SCIPY_FFT_DST:
        try:
            y = _sp_fft.dst(a, type=2, axis=-1, norm=None, workers=-1)  # type: ignore[arg-type]
        except TypeError:
            try:
                y = _sp_fft.dst(a, type=2, axis=-1, norm=None)  # type: ignore[arg-type]
            except TypeError:
                # Very old SciPy: no norm kw; fallback to fftpack below
                y = None  # type: ignore[assignment]
        if y is not None:  # type: ignore[truthy-bool]
            try:
                y = _sp_fft.dst(y, type=2, axis=-2, norm=None, workers=-1)  # type: ignore[arg-type]
            except TypeError:
                y = _sp_fft.dst(y, type=2, axis=-2, norm=None)  # type: ignore[arg-type]
            return y
    # Fallback to fftpack.dst
    if _sp_fftpack is not None and _HAS_FFTPACK_DST:
        y = _sp_fftpack.dst(a, type=2, axis=-1)  # type: ignore[arg-type]
        y = _sp_fftpack.dst(y, type=2, axis=-2)  # type: ignore[arg-type]
        return y
    # Last-resort numpy fallback for 2D (not optimized)
    return _dstn_numpy_fallback(a)

def _dstn_fast(a: np.ndarray) -> np.ndarray:
    """
    Use preselected fastest dstn implementation if available, else fallback.
    """
    if _DSTN_IMPL is not None:
        return _DSTN_IMPL(a)  # type: ignore[misc]
    if _sp_fftpack is not None and _HAS_FFTPACK_DSTN:
        return _sp_fftpack.dstn(a, type=2)  # type: ignore[arg-type]
    return _dstn_numpy_fallback(a)
def _dstn_numpy_fallback(a: np.ndarray) -> np.ndarray:
    """
    Minimal numpy-only fallback for 2D DST-II (un-normalized).
    """
    if a.ndim != 2:
        raise ValueError("Numpy fallback only supports 2D arrays.")
    m, n = a.shape
    jm = np.arange(m, dtype=np.float64).reshape(-1, 1)
    im = np.arange(m, dtype=np.float64).reshape(1, -1)
    jn = np.arange(n, dtype=np.float64).reshape(-1, 1)
    in_ = np.arange(n, dtype=np.float64).reshape(1, -1)

    Sm = np.sin(np.pi * (im + 0.5) * (jm + 1.0) / float(m))
    Sn = np.sin(np.pi * (in_ + 0.5) * (jn + 1.0) / float(n))

    y = Sm @ a @ Sn.T
    return y

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute the 2D DST Type II for an n×n real-valued array.

        Uses scipy.fft.dst (two passes along axes) when available (fast, multithreaded),
        falling back to scipy.fft.dstn or scipy.fftpack, and finally a small numpy-based
        2D fallback.
        """
        # Use asarray to avoid unnecessary copies; downstream routines will allocate outputs.
        a = np.asarray(problem)

        return _dstn_fast(a)
        return result