from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple, Union
import numpy as np
from scipy import signal as _signal

ArrayLike1D = Union[Sequence[float], np.ndarray]
Pair = Tuple[ArrayLike1D, ArrayLike1D]
Batch = Sequence[Pair]

def _should_use_fft(n_a: int, n_b: int) -> bool:
    # Heuristic: favor direct for most cases; switch to FFT only when it's clearly better.
    # direct ~ n_a * n_b
    # fft ~ C * n * log2(n), n = n_a + n_b - 1
    na, nb = n_a, n_b
    if na == 0 or nb == 0:
        return False
    direct_cost = na * nb
    # If the direct path is small enough, it will be very fast in C.
    if direct_cost <= 3_000_000:
        return False
    n = na + nb - 1
    if n <= 1:
        return False
    fft_cost = n * (np.log2(n))
    # Use a large fudge factor to switch only when FFT is much better.
    return (32.0 * fft_cost) < direct_cost

def _convolve1d(a: np.ndarray, b: np.ndarray, mode: str = "full") -> np.ndarray:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")
    mode = mode.lower()
    if mode not in ("full", "same", "valid"):
        raise ValueError(f"Unsupported mode: {mode}. Supported: 'full', 'same', 'valid'.")

    # Use direct convolution for most cases; switch to FFT for very large inputs.
    if _should_use_fft(a.size, b.size):
        # SciPy's fftconvolve uses next_fast_len internally and is highly optimized.
        return _signal.fftconvolve(a, b, mode=mode)
    else:
        return np.convolve(a, b, mode=mode)

def _is_numeric_1d_sequence(x: Any) -> bool:
    # Determine if x represents a 1D numeric array-like (list/tuple/ndarray of numbers),
    # not a pair (a, b) itself.
    if isinstance(x, np.ndarray):
        return x.ndim == 1 and np.issubdtype(x.dtype, np.number)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return True
        first = x[0]
        # If the first element is itself array-like, then x is likely a pair (a, b), not 1D data.
        return not isinstance(first, (list, tuple, np.ndarray))
    return False

class Solver:
    def __init__(self, mode: str = "full") -> None:
        self.mode = mode

    def _solve_single(self, problem: Pair, mode: str) -> np.ndarray:
        a, b = problem
        a_np = np.asarray(a)
        b_np = np.asarray(b)
        # Ensure numeric dtype
        if not (np.issubdtype(a_np.dtype, np.number) and np.issubdtype(b_np.dtype, np.number)):
            a_np = a_np.astype(float, copy=False)
            b_np = b_np.astype(float, copy=False)
        return _convolve1d(a_np, b_np, mode=mode)

    def solve(self, problem: Union[Pair, Batch], **kwargs) -> Any:
        """
        Compute 1D convolution(s) for given input.

        - If 'problem' is a tuple (a, b), returns a single 1D numpy array.
        - If 'problem' is a sequence of tuples [(a1, b1), (a2, b2), ...],
          returns a list of 1D numpy arrays, one per pair.

        Optional kwargs:
        - mode: 'full' (default), 'same', or 'valid'
        """
        mode = kwargs.get("mode", getattr(self, "mode", "full"))

        # Detect single pair vs batch of pairs robustly
        if isinstance(problem, (tuple, list)) and len(problem) == 2:
            first, second = problem  # type: ignore[index]
            if _is_numeric_1d_sequence(first) and _is_numeric_1d_sequence(second):
                return self._solve_single(problem, mode=mode)

        # Otherwise treat as batch
        if not isinstance(problem, Iterable):
            raise ValueError("Problem must be a pair (a, b) or a sequence of such pairs.")
        outputs: List[np.ndarray] = []
        for item in problem:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each item in batch must be a tuple/list of (a, b).")
            outputs.append(self._solve_single(item, mode=mode))
        return outputs