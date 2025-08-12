import numpy as np
from typing import List, Tuple, Any
from scipy import signal

class Solver:
    def solve(self, problem: List[Tuple[Any, Any, int, int]]) -> List[np.ndarray]:
        """
        Perform upsample, FIR filter, and downsample (upfirdn) for each problem instance.

        Parameters
        ----------
        problem : List[Tuple[h, x, up, down]]
            Each tuple contains:
                h   : 1‑D array-like filter coefficients
                x   : 1‑D array-like input signal
                up  : integer up‑sampling factor (>=1)
                down: integer down‑sampling factor (>=1)

        Returns
        -------
        List[np.ndarray]
            List of filtered and resampled 1‑D arrays.
        """
        results = []
        for h, x, up, down in problem:
            # Use SciPy's upfirdn to ensure exact reference behavior
            res = signal.upfirdn(h, x, up=up, down=down)
            results.append(np.asarray(res, dtype=np.float64))
        return results