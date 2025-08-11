import numpy as np
from typing import Any, Tuple

class Solver:
    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        """
        Compute the 2D correlation of arrays a and b using "full" mode with "fill" boundary.
        Implemented via FFT-based convolution (correlation = convolution with flipped kernel).
        Uses real-valued FFTs and selects fast FFT sizes when available for speed.
        """
        a, b = problem

        # Convert to numpy arrays with float64 for accuracy
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both inputs must be 2D arrays.")

        M, N = a.shape
        P, Q = b.shape

        # Output shape for 'full' correlation
        out_M = M + P - 1
        out_N = N + Q - 1

        # Flip kernel for correlation -> convolution equivalence
        b = b[::-1, ::-1]

        # Try to use numpy's next_fast_len if available, else fallback
        try:
            from numpy.fft import next_fast_len  # type: ignore
            fM = next_fast_len(out_M)
            fN = next_fast_len(out_N)
        except Exception:
            # Fallback: find a nearby 2,3,5-smooth length or next power of two
            def _is_235_smooth(x: int) -> bool:
                if x <= 0:
                    return False
                for p in (2, 3, 5):
                    while x % p == 0 and x > 1:
                        x //= p
                return x == 1

            def _next_fast_len(n: int) -> int:
                m = max(1, int(n))
                # limit search to avoid long loops; if not found, use next power of two
                limit = m + 4096
                while m <= limit:
                    if _is_235_smooth(m):
                        return m
                    m += 1
                return 1 << ((int(n) - 1).bit_length())

            fM = _next_fast_len(out_M)
            fN = _next_fast_len(out_N)

        # Ensure contiguous arrays for better FFT performance
        a_c = np.ascontiguousarray(a)
        b_c = np.ascontiguousarray(b)

        # Compute real FFTs with padding to (fM, fN)
        # Use rfftn/irfftn for broad compatibility
        fa = np.fft.rfftn(a_c, s=(fM, fN))
        fb = np.fft.rfftn(b_c, s=(fM, fN))

        # Point-wise multiply in frequency domain and invert
        fc = fa * fb
        conv = np.fft.irfftn(fc, s=(fM, fN))

        # Truncate to the required 'full' correlation size
        result = conv[:out_M, :out_N]

        return result