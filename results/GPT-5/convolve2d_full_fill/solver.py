from typing import Any, Tuple

import math
import numpy as np

class Solver:
    """
    Efficient solver for 2D full convolution with zero (fill) boundary.
    Uses FFT-based convolution (real FFT when possible) for speed.
    """

    @staticmethod
    def _next_fast_len_235(n: int) -> int:
        """Return the smallest 2/3/5-smooth integer >= n."""
        if n <= 1:
            return 1
        # Upper bounds for exponents ensure we can reach >= n
        mi = int(math.ceil(math.log(n, 2))) + 1
        mj = int(math.ceil(math.log(n, 3))) + 1
        best = 1 << 62  # large sentinel
        for i in range(mi + 1):
            p2 = 1 << i
            if p2 >= best:
                break
            p3 = p2
            for _ in range(mj + 1):
                if p3 >= best:
                    break
                val = p3
                while val < n:
                    val *= 5
                if val < best:
                    best = val
                p3 *= 3
        # Also consider pure powers (could be smaller than combinations due to loop limits)
        # Next power of 2
        p2 = 1 << int(math.ceil(math.log(n, 2)))
        if p2 < n:
            p2 <<= 1
        best = min(best, p2)
        # Next power of 3
        p3 = 1
        while p3 < n:
            p3 *= 3
        best = min(best, p3)
        # Next power of 5
        p5 = 1
        while p5 < n:
            p5 *= 5
        best = min(best, p5)
        return int(best)

    @staticmethod
    def _fftconvolve2d_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Ensure inputs are numpy arrays
        a = np.asarray(a)
        b = np.asarray(b)

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both input arrays must be 2D.")

        # Output size for 'full' convolution
        s0 = a.shape[0] + b.shape[0] - 1
        s1 = a.shape[1] + b.shape[1] - 1

        # Choose FFT sizes (2/3/5-smooth for speed)
        if s0 == s1:
            p = Solver._next_fast_len_235(s0)
            s = (p, p)
        else:
            p0 = Solver._next_fast_len_235(s0)
            p1 = Solver._next_fast_len_235(s1)
            s = (p0, p1)

        # Determine computation path based on dtype
        if np.iscomplexobj(a) or np.iscomplexobj(b):
            # Complex convolution
            dtype = np.result_type(a.dtype, b.dtype, np.complex128)
            a_c = np.ascontiguousarray(a, dtype=dtype)
            b_c = np.ascontiguousarray(b, dtype=dtype)
            Fa = np.fft.fft2(a_c, s)
            Fb = np.fft.fft2(b_c, s)
            Fa *= Fb  # in-place multiply to reduce memory
            del Fb
            out = np.fft.ifft2(Fa)
            out = out[:s0, :s1]
            # Keep complex dtype if complex inputs present
            return out.astype(dtype, copy=False)
        else:
            # Real convolution
            dtype = np.result_type(a.dtype, b.dtype, np.float64)
            a_c = np.ascontiguousarray(a, dtype=dtype)
            b_c = np.ascontiguousarray(b, dtype=dtype)
            Fa = np.fft.rfft2(a_c, s)
            Fb = np.fft.rfft2(b_c, s)
            Fa *= Fb  # in-place multiply
            del Fb
            out = np.fft.irfft2(Fa, s)
            out = out[:s0, :s1]
            return out.astype(dtype, copy=False)

    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        """
        Compute the 2D convolution of arrays a and b using "full" mode and "fill" boundary.

        :param problem: A tuple (a, b) of 2D arrays.
        :return: A 2D array containing the convolution result.
        """
        a, b = problem
        return self._fftconvolve2d_full(a, b)