import numpy as np
from scipy import signal
from typing import List, Tuple, Any
class Solver:
    def __init__(self, mode: str = "full"):
        """
        Parameters
        ----------
        mode : str, optional
            Convolution mode. Either "full" (default) or "valid".
            For "valid", the second array must not be longer than the first.
        """
        if mode not in ("full", "valid"):
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode

    def solve(self, problem, **kwargs):
        """
        Compute 1‑D correlations for a list of array pairs or a single pair.

        Parameters
        ----------
        problem : List[Tuple[List[float], List[float]]] or Tuple[List[float], List[float]]
            Either a list of pairs ``(a, b)`` or a single pair.

        Returns
        -------
        List[np.ndarray] or np.ndarray
            Correlation results. A list is returned for a list input,
            otherwise a single ``np.ndarray`` is returned.
        """
        # Normalise input: make it iterable of pairs
        if isinstance(problem, tuple) and len(problem) == 2:
            # single pair – wrap in a list for unified processing
            pairs = [problem]
            single_output = True
        else:
            # assume an iterable of pairs
            pairs = list(problem)
            single_output = False

        results = []
        for a, b in pairs:
            # Convert to NumPy arrays (ensures fast vectorised operations)
            a_arr = np.asarray(a, dtype=float)
            b_arr = np.asarray(b, dtype=float)

            # Use SciPy's highly‑optimised convolution implementation.
            # It automatically selects the best algorithm (direct or FFT)
            # and correctly handles both "full" and "valid" modes.
            conv = signal.convolve(a_arr, b_arr, mode=self.mode)
            results.append(conv)

        # Return a single ndarray if the original input was a single pair
        return results[0] if single_output else results