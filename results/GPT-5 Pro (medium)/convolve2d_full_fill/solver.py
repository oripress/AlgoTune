from typing import Any, Tuple

import numpy as np

try:
    # Prefer SciPy's FFT (generally faster than NumPy's)
    from scipy import fft as sfft
except Exception:  # pragma: no cover
    sfft = None

class Solver:
    def __init__(self):
        # Fixed to match the problem statement and reference solver
        self.mode = "full"
        self.boundary = "fill"

    def _fft_module(self):
        # Choose the fastest available FFT backend
        if sfft is not None:
            return sfft.rfftn, sfft.irfftn
        # Fallback to NumPy's FFT
        return np.fft.rfftn, np.fft.irfftn

    def _next_fast_shape(self, shape: Tuple[int, int]) -> Tuple[int, int]:
        # Use SciPy's next_fast_len if available; otherwise, no padding beyond required
        if sfft is not None and hasattr(sfft, "next_fast_len"):
            return (sfft.next_fast_len(shape[0]), sfft.next_fast_len(shape[1]))
        return shape

    @staticmethod
    def _fft_call(func, arr, s):
        # Use multithreading if supported; gracefully fall back if not
        try:
            return func(arr, s=s, workers=-1)
        except TypeError:
            return func(arr, s=s)

    def solve(self, problem: Tuple[np.ndarray, np.ndarray], **kwargs) -> Any:
        """
        Compute the 2D convolution of arrays a and b using "full" mode and "fill" boundary.

        This uses FFT-based convolution for speed:
          y = irfftn(rfftn(a, S) * rfftn(b, S), S)[:out0, :out1]
        where out = (a0 + b0 - 1, a1 + b1 - 1) and S is an FFT-friendly padded size.
        """
        a, b = problem
        a = np.asarray(a)
        b = np.asarray(b)

        # Ensure 2D inputs
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("Both inputs must be 2D arrays.")

        # Output shape for "full" convolution
        out_shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)

        # Use float64 for numerical stability and to closely match SciPy's reference
        a64 = np.asarray(a, dtype=np.float64, order="C")
        b64 = np.asarray(b, dtype=np.float64, order="C")

        rfftn, irfftn = self._fft_module()

        # Choose FFT-friendly padded shape (improves performance significantly)
        pad_shape = self._next_fast_shape(out_shape)

        # FFT-based linear convolution with zero-padding ("fill" boundary)
        Fa = self._fft_call(rfftn, a64, s=pad_shape)
        Fb = self._fft_call(rfftn, b64, s=pad_shape)
        Fa *= Fb  # in-place multiply to reduce memory and allocations
        out_full = self._fft_call(irfftn, Fa, s=pad_shape)

        # Slice back to exact "full" convolution shape
        out = out_full[: out_shape[0], : out_shape[1]]

        if not np.isfinite(out).all():
            raise FloatingPointError("Non-finite values encountered in convolution result.")

        return out