import numpy as np
from scipy import signal

class Solver:
    def solve(self, problem: list, **kwargs) -> list:
        results = []
        for h, x, up, down in problem:
            h_arr = np.asarray(h, dtype=np.float64)
            x_arr = np.asarray(x, dtype=np.float64)
            res = signal.upfirdn(h_arr, x_arr, up=int(up), down=int(down))
            results.append(res)
        return results