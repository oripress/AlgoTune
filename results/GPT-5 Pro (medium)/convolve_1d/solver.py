from typing import Any, Tuple

import math
import numpy as np

class Solver:
    def __init__(self, mode: str = "full") -> None:
        # Default mode can be overridden by the evaluation harness
        self.mode = mode

    def _should_use_fft(self, n: int, m: int, dtype_a: np.dtype, dtype_b: np.dtype) -> bool:
        """
        Heuristic to decide when to use FFT-based convolution.

        We compare the naive O(n*m) cost to an O((n+m) log(n+m)) estimate with a safety factor.
        """
        conv_len = n + m - 1
        # Avoid FFT overhead for tiny problems
        if conv_len <= 64:
            return False

        direct_ops = n * m
        fft_ops = conv_len * max(1.0, math.log2(conv_len))

        # Adjust safety factor based on dtype complexity
        is_complex = np.issubdtype(np.result_type(dtype_a, dtype_b), np.complexfloating)
        # Larger factor means we require a stronger advantage to use FFT
        k = 28.0 if not is_complex else 8.0

        # If one operand is extremely small, naive may still be better unless the other is huge
        # This ratio check covers both balanced and imbalanced sizes.
        return direct_ops > k * fft_ops

    @staticmethod
    def _next_pow2(n: int) -> int:
        """Return the next power-of-two >= n."""
        return 1 << (n - 1).bit_length()

    def _fft_convolve_full(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute full 1D convolution via FFT. Returns length n+m-1.
        Assumes a and b are already converted to a floating or complex dtype.
        """
        n = a.size
        m = b.size
        conv_len = n + m - 1
        nfft = self._next_pow2(conv_len)

        if np.iscomplexobj(a) or np.iscomplexobj(b):
            A = np.fft.fft(a, nfft)
            B = np.fft.fft(b, nfft)
            y = np.fft.ifft(A * B, nfft)
            y = y[:conv_len]
            return y
        else:
            # Real-input FFT path
            A = np.fft.rfft(a, nfft)
            B = np.fft.rfft(b, nfft)
            y = np.fft.irfft(A * B, nfft)
            y = y[:conv_len]
            # Ensure exactly real dtype
            return y

    @staticmethod
    def _extract_mode(y: np.ndarray, n: int, m: int, mode: str) -> np.ndarray:
        """Extract the requested mode from the full convolution result."""
        mode = mode.lower()
        if mode == "full":
            return y
        if mode == "same":
            start = (m - 1) // 2
            return y[start : start + n]
        if mode == "valid":
            start = (m - 1) if n >= m else (n - 1)
            length = abs(n - m) + 1
            return y[start : start + length]
        # Fallback to full if an unknown mode is provided
        return y

    def solve(self, problem, **kwargs) -> Any:
        """
        Compute 1D convolution (used as correlation in the task) for a pair of 1D arrays.

        The implementation adapts between direct convolution (numpy.convolve) and an FFT-based
        approach for large inputs to maximize performance while matching SciPy's output.
        """
        a, b = problem  # Expect a tuple/list of 1D arrays
        # Determine mode priority: kwargs overrides instance attribute, default to 'full'
        mode = kwargs.get("mode", getattr(self, "mode", "full"))

        # Convert inputs to arrays and promote dtype to a safe computation type (float64/complex128)
        # This mirrors SciPy behavior for integer inputs and ensures numerical stability.
        compute_dtype = np.result_type(np.asarray(a).dtype, np.asarray(b).dtype, np.float64)
        a_arr = np.asarray(a, dtype=compute_dtype)
        b_arr = np.asarray(b, dtype=compute_dtype)

        n = a_arr.size
        m = b_arr.size

        # Handle empty inputs gracefully (align with numpy/scipy behavior by returning empty)
        if n == 0 or m == 0:
            return np.array([], dtype=compute_dtype)

        # Decide on method
        if self._should_use_fft(n, m, a_arr.dtype, b_arr.dtype):
            full = self._fft_convolve_full(a_arr, b_arr)
            # If inputs are real, ensure a real output precisely
            if not (np.iscomplexobj(a_arr) or np.iscomplexobj(b_arr)):
                full = np.real(full)
            return self._extract_mode(full, n, m, mode)
        else:
            # Direct method via NumPy's highly-optimized 1D convolution
            return np.convolve(a_arr, b_arr, mode=mode)