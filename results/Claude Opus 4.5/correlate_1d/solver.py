import numpy as np
from scipy.signal import fftconvolve

class Solver:
    def solve(self, problem, **kwargs):
        """Compute 1D correlation for each pair in the problem list."""
        results = []
        for pair in problem:
            a, b = pair[0], pair[1]
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            
            len_a, len_b = len(a), len(b)
            
            # For very small arrays, use direct correlation
            if len_a * len_b < 100:
                res = np.correlate(a, b, mode='full')
            else:
                # correlate(a, b) = convolve(a, b[::-1])
                res = fftconvolve(a, b[::-1], mode='full')
            results.append(res)
        return results