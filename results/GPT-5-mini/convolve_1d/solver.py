import numpy as np
from typing import Any, List, Sequence
from scipy.signal import fftconvolve

class Solver:
    """
    Simple, efficient 1D convolution solver.

    - Accepts either a single pair (a, b) or a list of pairs [(a1, b1), ...].
    - Uses numpy.convolve for small problems and scipy.signal.fftconvolve for larger ones.
    - Preserves complex inputs and supports modes: 'full', 'same', 'valid'.
    """

    def __init__(self, mode: str = "full", fft_thresh: int = 256):
        self.mode = mode
        # Threshold on full output length (n+m-1) to switch to FFT-based method
        self.fft_thresh = int(fft_thresh)

    def solve(self, problem, **kwargs) -> Any:
        mode = kwargs.get("mode", self.mode)

        if self._is_list_of_pairs(problem):
            out: List[np.ndarray] = []
            for pair in problem:
                a, b = pair
                out.append(self._conv_pair(a, b, mode))
            return out
        else:
            a, b = problem
            return self._conv_pair(a, b, mode)

    def _conv_pair(self, a, b, mode: str) -> np.ndarray:
        A = np.asarray(a)
        B = np.asarray(b)
        if A.ndim != 1:
            A = A.ravel()
        if B.ndim != 1:
            B = B.ravel()

        # Preserve complex dtype if any input is complex
        is_complex = np.iscomplexobj(A) or np.iscomplexobj(B)
        dtype = np.complex128 if is_complex else np.float64

        n = A.size
        m = B.size

        # Empty/trivial cases
        if n == 0 or m == 0:
            return np.array([], dtype=dtype)

        # 'valid' mode with second longer than first yields empty
        if mode == "valid" and m > n:
            return np.array([], dtype=dtype)

        full_len = n + m - 1

        # Choose method: direct for small sizes, FFT for larger sizes
        if full_len >= self.fft_thresh:
            # fftconvolve handles real/complex and modes correctly
            res = fftconvolve(A, B, mode=mode)
            # Ensure dtype consistency
            if not is_complex and np.iscomplexobj(res):
                # small imaginary noise may appear; cast to real
                res = res.real
            return np.asarray(res, dtype=dtype)
        else:
            return np.asarray(np.convolve(A, B, mode=mode), dtype=dtype)

    # -------------------------
    # Helper / internal methods
    # -------------------------
    @staticmethod
    def _is_sequence(x) -> bool:
        return isinstance(x, (list, tuple, np.ndarray))

    def _is_list_of_pairs(self, problem) -> bool:
        if not self._is_sequence(problem):
            return False
        if len(problem) == 0:
            return False
        first = problem[0]
        if not self._is_sequence(first):
            return False
        if len(first) != 2:
            return False
        return self._is_sequence(first[0])