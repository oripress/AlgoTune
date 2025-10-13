from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import math
import numpy as np


def _correlate1d_fft(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    """
    FFT-based cross-correlation of 1D real arrays.
    Correlation is computed as convolution of a with reversed b.
    """
    n = a.shape[0]
    m = b.shape[0]
    L = n + m - 1

    # Reverse b for correlation
    b_rev = b[::-1]

    # Next power of two for efficient FFT
    nfft = 1 << (L - 1).bit_length()

    # Use real FFTs
    fa = np.fft.rfft(a, nfft)
    fb = np.fft.rfft(b_rev, nfft)
    c_full = np.fft.irfft(fa * fb, nfft)[:L]

    if mode == "full":
        return c_full
    elif mode == "same":
        S = max(n, m)
        start = (L - S) // 2
        return c_full[start : start + S]
    elif mode == "valid":
        start = min(n, m) - 1
        end = max(n, m) - 1
        return c_full[start : end + 1]
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _should_use_fft(n: int, m: int) -> bool:
    """
    Heuristic to decide between direct (np.correlate) and FFT-based method.

    Compares approximate operation counts:
    - Direct: O(n*m)
    - FFT:    O(L log L), L = n + m - 1
    Includes a safety factor to bias toward the generally faster direct method
    for small to moderate sizes, switching to FFT only when clearly beneficial.
    """
    L = n + m - 1
    if L <= 64:
        return False  # tiny sizes: direct is best
    # Approximate comparisons (safety factors tuned empirically)
    direct_cost = n * m
    fft_cost = 6.0 * L * math.log2(L)  # factor accounts for multiple FFTs and constants
    # Also ensure we only switch for sufficiently large problems to amortize FFT overhead
    return (direct_cost > fft_cost) and (L >= 512)


def _correlate1d(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    """
    Cross-correlation wrapper that dynamically chooses the fastest method.
    """
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        # Match NumPy/SciPy behavior: empty input leads to empty output
        # For correlate, NumPy raises ValueError on empty inputs; SciPy too.
        # To be robust, return empty array.
        return np.array([], dtype=np.float64)

    # Prefer direct method for small/medium sizes
    if not _should_use_fft(n, m):
        # NumPy's correlate computes cross-correlation directly
        return np.correlate(a, b, mode)

    # Use FFT-based method for large cases
    return _correlate1d_fft(a, b, mode)


class Solver:
    def __init__(self) -> None:
        # Default mode; can be overridden via kwargs in solve or by setting self.mode externally
        self.mode: str = "full"

    def solve(self, problem: Sequence[Tuple[Iterable[float], Iterable[float]]], **kwargs) -> List[np.ndarray]:
        """
        Compute the 1D correlation for each pair in the problem list.

        Supports modes 'full', 'same', and 'valid'.
        If 'mode' is provided via kwargs, it overrides the default/self.mode.

        :param problem: Sequence of tuples (a, b), each a 1D array-like of floats.
        :param kwargs: Optional parameters, especially 'mode'.
        :return: List of 1D NumPy arrays with the correlation results.
        """
        mode = kwargs.get("mode", getattr(self, "mode", "full"))
        # Normalize mode to a small set to avoid overhead of repeated checks
        if mode not in ("full", "same", "valid"):
            # Default to 'full' if unspecified or invalid
            mode = "full"
        # Persist mode for potential subsequent calls
        self.mode = mode

        results: List[np.ndarray] = []
        append = results.append

        for pair in problem:
            a, b = pair
            # Convert to contiguous float64 arrays for numerical stability and performance
            a_arr = np.asarray(a, dtype=np.float64)
            b_arr = np.asarray(b, dtype=np.float64)

            # Ensure 1D
            if a_arr.ndim != 1 or b_arr.ndim != 1:
                a_arr = a_arr.ravel()
                b_arr = b_arr.ravel()

            res = _correlate1d(a_arr, b_arr, mode)
            # Ensure finite values
            if not np.all(np.isfinite(res)):
                # Fall back to direct method if something odd happened (e.g., numerical issues)
                res = np.correlate(a_arr, b_arr, mode)
            append(res)

        return results