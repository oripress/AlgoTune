import numpy as np
from scipy.signal import upfirdn

class Solver:
    def solve(self, problem, **kwargs):
        results = []
        for h, x, up, down in problem:
            # Ensure arrays are numeric
            h_arr = np.asarray(h, dtype=np.float64)
            x_arr = np.asarray(x, dtype=np.float64)
            res = upfirdn(h_arr, x_arr, up=up, down=down)
            results.append(res)
        return results