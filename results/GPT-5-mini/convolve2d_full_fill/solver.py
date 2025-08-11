import numpy as np
from typing import Any, Tuple

# Try to use scipy.fft (pocketfft) for faster FFTs and next_fast_len; fall back to numpy.fft.
try:
    from scipy.fft import rfft2 as _rfft2, irfft2 as _irfft2, next_fast_len as _next_fast_len  # type: ignore
except Exception:
    try:
        from numpy.fft import rfft2 as _rfft2, irfft2 as _irfft2  # type: ignore
    except Exception:
        _rfft2 = np.fft.rfft2
        _irfft2 = np.fft.irfft2
    _next_fast_len = None  # type: ignore

# Use scipy.signal.convolve2d for small sizes if available (optimized C implementation).
try:
    from scipy.signal import convolve2d as _convolve2d  # type: ignore
except Exception:
    _convolve2d = None  # type: ignore

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        """
        Compute 2D convolution in 'full' mode with 'fill' (zero) boundary handling.

        Uses:
        - scipy.signal.convolve2d for small problems (when available).
        - FFT-based convolution (rfft2/irfft2) for larger problems, with padded sizes
          chosen via scipy.fft.next_fast_len when available, else next power of two.
        """
        a, b = problem

        # Convert inputs to contiguous float64 arrays
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        b = np.ascontiguousarray(np.asarray(b, dtype=np.float64))

        # Ensure 2D arrays
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a.reshape(1, -1)
        elif a.ndim > 2:
            a = a.reshape(a.shape[0], -1)

        if b.ndim == 0:
            b = b.reshape(1, 1)
        elif b.ndim == 1:
            b = b.reshape(1, -1)
        elif b.ndim > 2:
            b = b.reshape(b.shape[0], -1)

        ha, wa = a.shape
        hb, wb = b.shape
        out_h = ha + hb - 1
        out_w = wa + wb - 1

        # Degenerate output
        if out_h <= 0 or out_w <= 0:
            return np.zeros((max(0, out_h), max(0, out_w)), dtype=np.float64)

        # Heuristic: use direct optimized convolution for small problems to avoid FFT overhead
        # Threshold tuned to avoid expensive FFT overhead on small inputs.
        if (_convolve2d is not None) and (a.size * b.size <= 1_000_000):
            try:
                res = _convolve2d(a, b, mode="full", boundary="fill")
                return np.ascontiguousarray(res, dtype=np.float64)
            except Exception:
                # Fall back to FFT approach on any failure
                pass

        # Helper: next power of two
        def _next_pow2(n: int) -> int:
            if n <= 1:
                return 1
            return 1 << ((n - 1).bit_length())

        # Choose padded FFT size using next_fast_len if available, else next power of two
        def _choose_size(n: int) -> int:
            if _next_fast_len is not None:
                try:
                    v = int(_next_fast_len(int(n)))
                    if v >= n:
                        return v
                except Exception:
                    pass
            return _next_pow2(n)

        f0 = _choose_size(out_h)
        f1 = _choose_size(out_w)

        # Compute real 2D FFTs with zero-padding to (f0, f1)
        Fa = _rfft2(a, s=(f0, f1))
        Fb = _rfft2(b, s=(f0, f1))

        # Multiply in frequency domain in-place to save memory and then invert
        try:
            # Use out parameter to avoid allocating an intermediate array
            np.multiply(Fa, Fb, out=Fa)
            conv = _irfft2(Fa, s=(f0, f1))
        except Exception:
            # Fallback to safe allocation if in-place multiply fails for any reason
            Fc = Fa * Fb
            conv = _irfft2(Fc, s=(f0, f1))

        # Crop to the 'full' convolution size and return
        result = conv[:out_h, :out_w]
        return np.ascontiguousarray(result, dtype=np.float64)