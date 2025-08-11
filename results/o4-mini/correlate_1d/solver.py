import numpy as np
import os
from typing import Any, List
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import correlate as sp_correlate

class Solver:
    """
    Solver for 1D correlation of list of pairs of 1D arrays.
    """

    def solve(self, problem: List, **kwargs) -> List:
        """
        Compute the 1D correlation for each valid pair in the problem list.

        Args:
            problem: List of tuples/lists/arrays (a, b), 1D numeric sequences.
            mode: Correlation mode, 'full' or 'valid'. Default is 'full'.

        Returns:
            List of numpy arrays containing correlation results.
        """
        mode = kwargs.get('mode', getattr(self, 'mode', 'full'))
        arrs = []
        brrs = []
        for a, b in problem:
            arr = np.asarray(a, dtype=np.float64)
            brr = np.asarray(b, dtype=np.float64)
            if mode == 'valid' and brr.size > arr.size:
                continue
            arrs.append(arr)
            brrs.append(brr)

        n = len(arrs)
        if n == 0:
            return []

        def corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return sp_correlate(a, b, mode)

        cpu = os.cpu_count() or 1
        if n >= 2 and cpu > 1:
            workers = min(n, cpu)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(corr, arrs, brrs))
        else:
            results = [corr(a, b) for a, b in zip(arrs, brrs)]

        return results