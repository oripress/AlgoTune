import numpy as np
import scipy.fft
from numpy.typing import NDArray

class Solver:
    def solve(self, problem: NDArray) -> NDArray:
        """
        Compute the N-dimensional DST Type II using scipy.fft.
        """
        # Use all available workers and allow overwriting input for speed
        # We copy the input because overwrite_x might modify the original array
        # and we don't know if the caller expects it to be preserved.
        # However, copying takes time. If we can overwrite, it's faster.
        # Let's try overwriting a copy first.
        
        # Actually, scipy.fft.dstn might not support overwrite_x for all backends.
        # But let's try.
        
        # Also, ensure we are using float32 or float64 appropriately.
        # The input is real-valued.
        
        # Heuristic for workers
        workers = -1 if problem.size > 10000 else 1
        result = scipy.fft.dstn(problem.astype(np.float32, copy=False), type=2, workers=workers, overwrite_x=True)
        return result