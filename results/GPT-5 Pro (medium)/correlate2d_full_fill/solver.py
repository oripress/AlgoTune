from typing import Any, Tuple

import numpy as np

def _next_fast_len_5smooth(n: int) -> int:
    """Return the smallest 5-smooth number >= n (good for FFT performance)."""
    if n <= 2:
        return n
    # Start from n and search upward for a number whose prime factors are only 2,3,5
    m = n
    while True:
        x = m
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 1

class Solver:
    def __init__(self) -> None:
        # For reference compatibility; not strictly used by our implementation
        self.mode = "full"
        self.boundary = "fill"

    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        """
        Compute the 2D correlation of arrays a and b using "full" mode and "fill" boundary.
        This implementation uses FFT-based convolution with a reversed kernel to exactly
        match correlate2d semantics and alignment, with 5-smooth FFT sizes for speed.

        :param problem: A tuple (a, b) of 2D arrays.
        :return: A 2D array containing the correlation result.
        """
        a, b = problem
        a = np.asarray(a)
        b = np.asarray(b)

        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Inputs must be 2D arrays")

        # Output shape for 'full' correlation
        out0 = a.shape[0] + b.shape[0] - 1
        out1 = a.shape[1] + b.shape[1] - 1
        out_shape = (out0, out1)

        # Use double precision for numerical stability and tolerance adherence
        a64 = a.astype(np.float64, copy=False)
        b64 = b.astype(np.float64, copy=False)

        # Correlation equals convolution with both axes reversed on b
        # We compute via FFT with zero-padding to a 5-smooth length for speed.
        fft0 = _next_fast_len_5smooth(out0)
        fft1 = _next_fast_len_5smooth(out1)

        Fa = np.fft.rfftn(a64, s=(fft0, fft1))
        # Reverse both axes of b for correlation
        Fb_rev = np.fft.rfftn(b64[::-1, ::-1], s=(fft0, fft1))
        Fc = Fa * Fb_rev
        result_full = np.fft.irfftn(Fc, s=(fft0, fft1))

        # Crop to the exact 'full' output size
        result = result_full[:out0, :out1]

        # Ensure real output (numerical noise from FFT may introduce tiny imaginary parts)
        if np.iscomplexobj(result):
            result = result.real

        return result