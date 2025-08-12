import numpy as np
from typing import Any, Dict

class Solver:
    def __init__(self):
        # Cache numpy for speed
        self.np = np

        # Pre-import fftconvolve to avoid import overhead in solve (init time not counted)
        try:
            from scipy.signal import fftconvolve as _fftconvolve  # type: ignore
            self.fftconvolve = _fftconvolve
        except Exception:
            # Fallback: implement a lightweight FFT-based convolution using numpy's rfft/irfft
            try:
                # Try to get next_fast_len if available
                try:
                    from numpy.fft import next_fast_len  # type: ignore
                    self.next_fast_len = next_fast_len
                except Exception:
                    def _next_fast_len(n: int) -> int:
                        if n <= 1:
                            return 1
                        return 1 << ((n - 1).bit_length())
                    self.next_fast_len = _next_fast_len

                def _fftconvolve_fallback(a, b, mode="full"):
                    a = np.asarray(a, dtype=np.float64)
                    b = np.asarray(b, dtype=np.float64)

                    nx = a.size
                    ny = b.size

                    if nx == 0 or ny == 0:
                        return np.array([], dtype=np.float64)

                    # Quick paths
                    if nx == 1:
                        full = (a[0] * b).astype(np.float64)
                    elif ny == 1:
                        full = (b[0] * a).astype(np.float64)
                    else:
                        full_len = nx + ny - 1
                        N = int(self.next_fast_len(full_len))
                        # Use rfft/irfft for real inputs
                        A = np.fft.rfft(a, n=N)
                        B = np.fft.rfft(b, n=N)
                        C = A * B
                        conv = np.fft.irfft(C, n=N)
                        full = conv[:full_len].astype(np.float64)

                    if mode == "full":
                        return full
                    elif mode == "same":
                        out_len = nx
                        start = (full.size - out_len) // 2
                        return full[start:start + out_len]
                    elif mode == "valid":
                        out_len = max(0, max(nx, ny) - min(nx, ny) + 1)
                        if out_len == 0:
                            return np.array([], dtype=np.float64)
                        start = min(nx, ny) - 1
                        return full[start:start + out_len]
                    else:
                        return full

                self.fftconvolve = _fftconvolve_fallback
            except Exception:
                # Last resort: fallback to numpy.convolve with appropriate slicing
                def _convolve_simple(a, b, mode="full"):
                    a = np.asarray(a, dtype=np.float64)
                    b = np.asarray(b, dtype=np.float64)
                    nx = a.size
                    ny = b.size
                    if nx == 0 or ny == 0:
                        return np.array([], dtype=np.float64)
                    full = np.convolve(a, b).astype(np.float64)
                    if mode == "full":
                        return full
                    elif mode == "same":
                        out_len = nx
                        start = (full.size - out_len) // 2
                        return full[start:start + out_len]
                    elif mode == "valid":
                        out_len = max(0, max(nx, ny) - min(nx, ny) + 1)
                        if out_len == 0:
                            return np.array([], dtype=np.float64)
                        start = min(nx, ny) - 1
                        return full[start:start + out_len]
                    else:
                        return full
                self.fftconvolve = _convolve_simple

    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Compute 1-D convolution using the fastest available FFT implementation.
        Returns {"convolution": [...]} where the list contains floats.
        """
        x_in = problem.get("signal_x", [])
        y_in = problem.get("signal_y", [])
        mode = problem.get("mode", "full") or "full"
        mode = str(mode).lower()

        x = self.np.asarray(x_in, dtype=self.np.float64).ravel()
        y = self.np.asarray(y_in, dtype=self.np.float64).ravel()

        nx = x.size
        ny = y.size

        # Handle empty inputs
        if nx == 0 or ny == 0:
            return {"convolution": []}

        # Compute convolution using pre-imported implementation
        try:
            result = self.fftconvolve(x, y, mode=mode)
        except Exception:
            # Fallback robust path
            result = np.convolve(x, y)
            if mode != "full":
                full_len = nx + ny - 1
                if mode == "same":
                    start = (full_len - nx) // 2
                    result = result[start:start + nx]
                elif mode == "valid":
                    out_len = max(0, max(nx, ny) - min(nx, ny) + 1)
                    if out_len == 0:
                        return {"convolution": []}
                    start = min(nx, ny) - 1
                    result = result[start:start + out_len]

        result = np.asarray(result, dtype=np.float64)

        # If numerical issues occur (NaN/Inf), fallback to direct convolution
        if result.size > 0 and not np.all(np.isfinite(result)):
            direct = np.convolve(x, y).astype(np.float64)
            if mode == "full":
                result = direct
            elif mode == "same":
                start = (nx + ny - 1 - nx) // 2
                result = direct[start:start + nx]
            elif mode == "valid":
                out_len = max(0, max(nx, ny) - min(nx, ny) + 1)
                if out_len == 0:
                    return {"convolution": []}
                start = min(nx, ny) - 1
                result = direct[start:start + out_len]

        return {"convolution": result.tolist()}