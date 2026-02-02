import numpy as np
from scipy.signal import fftconvolve, oaconvolve

class Solver:
    def solve(self, problem, **kwargs):
        x = np.asarray(problem["signal_x"], dtype=np.float64)
        y = np.asarray(problem["signal_y"], dtype=np.float64)
        mode = problem.get("mode", "full")
        
        len_x, len_y = len(x), len(y)
        
        if len_x == 0 or len_y == 0:
            return {"convolution": np.array([])}
        
        min_len = min(len_x, len_y)
        max_len = max(len_x, len_y)
        
        # Use direct convolution for small signals
        if min_len <= 16:
            result = np.convolve(x, y, mode=mode)
        # Use overlap-add for very large signals with large ratio difference
        elif max_len > 10000 and max_len > 10 * min_len:
            result = oaconvolve(x, y, mode=mode)
        else:
            result = fftconvolve(x, y, mode=mode)
        
        return {"convolution": result}