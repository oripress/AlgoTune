from typing import Any

import numpy as np

def _next_pow_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << ((n - 1).bit_length())

def _convolve_full_fft(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.size
    m = y.size
    if n == 0 or m == 0:
        return np.empty(0, dtype=np.result_type(x.dtype, y.dtype))

    L = n + m - 1
    N = _next_pow_two(L)

    # Use real FFT if both inputs are real for speed
    if not np.iscomplexobj(x) and not np.iscomplexobj(y):
        Xf = np.fft.rfft(x, n=N)
        Yf = np.fft.rfft(y, n=N)
        Zf = Xf * Yf
        out = np.fft.irfft(Zf, n=N)[:L]
        return out
    else:
        Xf = np.fft.fft(x, n=N)
        Yf = np.fft.fft(y, n=N)
        Zf = Xf * Yf
        out = np.fft.ifft(Zf, n=N)[:L]
        return out

def _convolve_full_direct(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Use numpy's direct convolution for small sizes
    return np.convolve(x, y, mode="full")

def _trim_result(full: np.ndarray, n: int, m: int, mode: str) -> np.ndarray:
    if mode == "full":
        return full
    elif mode == "same":
        start = (m - 1) // 2
        end = start + n
        return full[start:end]
    elif mode == "valid":
        start = min(n, m) - 1
        end = max(n, m)
        return full[start:end]
    else:
        # Fallback to full if invalid mode provided
        return full

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Compute convolution using an FFT-based approach (with direct fallback for small inputs).
        Returns:
            {"convolution": list_of_values}
        """
        x = np.asarray(problem.get("signal_x", []))
        y = np.asarray(problem.get("signal_y", []))
        mode = problem.get("mode", "full")

        # Handle empty inputs early
        n = x.size
        m = y.size
        if n == 0 or m == 0:
            return {"convolution": []}

        # Promote dtype appropriately
        if np.iscomplexobj(x) or np.iscomplexobj(y):
            x = x.astype(np.complex128, copy=False)
            y = y.astype(np.complex128, copy=False)
        else:
            x = x.astype(np.float64, copy=False)
            y = y.astype(np.float64, copy=False)

        # Heuristic: direct convolution is faster for small problems
        if n * m <= 2048:
            full = _convolve_full_direct(x, y)
        else:
            full = _convolve_full_fft(x, y)

        result = _trim_result(full, n, m, mode)

        # Ensure real output if both inputs are real
        if not (np.iscomplexobj(x) or np.iscomplexobj(y)) and np.iscomplexobj(result):
            result = result.real

        return {"convolution": result.tolist()}