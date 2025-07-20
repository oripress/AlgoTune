import numpy as np
from scipy.fftpack import dst

class Solver:
    def solve(self, problem, **kwargs):
        """
        Compute 2D DST Type II via two in‐place FFTPACK DST-II passes:
        first along columns (axis=1), then along rows (axis=0).
        """
        # Ensure a contiguous, writeable float64 array
        arr = np.require(problem, dtype=np.float64, requirements=["C", "W"])
        # DST along columns
        arr = dst(arr, type=2, axis=1, overwrite_x=True)
        # DST along rows
        arr = dst(arr, type=2, axis=0, overwrite_x=True)
        return arr