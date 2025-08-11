from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
from scipy import signal

ArrayLike1D = Union[Sequence[float], np.ndarray]

class Solver:
    def __init__(self, mode: str = "full") -> None:
        # Default mode; can be overridden per-call via kwargs
        self.mode = mode

    def solve(self, problem: List[Tuple[ArrayLike1D, ArrayLike1D]], **kwargs) -> List[np.ndarray]:
        """
        Compute 1D cross-correlation for each pair in the problem list using SciPy's correlate.

        - Supports modes: 'full' (default), 'same', 'valid'
        """
        mode = kwargs.get("mode", getattr(self, "mode", "full"))
        results: List[np.ndarray] = []

        for a, b in problem:
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)

            # Ensure 1D
            if a_arr.ndim != 1:
                a_arr = a_arr.reshape(-1)
            if b_arr.ndim != 1:
                b_arr = b_arr.reshape(-1)

            res = signal.correlate(a_arr, b_arr, mode=mode)

            # Guard against NaN/Inf (inputs are expected finite, but ensure compliance)
            if not np.all(np.isfinite(res)):
                res = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

            results.append(res)

        return results