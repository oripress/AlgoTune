import numpy as np
from numba import njit

@njit(cache=True)
def cumulative_simpson_nb(y, dx):
    N = y.shape[0]
    # allocate result buffer
    res = np.empty(N, np.float64)
    if N == 0:
        return res
    res[0] = 0.0
    if N > 1:
        half_dx = dx * 0.5
        third_dx = dx / 3.0
        # first segment (trapezoid)
        res[1] = half_dx * (y[0] + y[1])
        i = 2
        # process two segments at a time: Simpson on even, trapezoid on odd
        while i < N:
            # Simpson's rule on two segments ending at i
            res[i] = res[i-2] + third_dx * (y[i-2] + 4.0 * y[i-1] + y[i])
            j = i + 1
            if j < N:
                # trapezoid on last segment
                res[j] = res[i] + half_dx * (y[i] + y[j])
            i += 2
    return res

# trigger compile at import time
_dummy = cumulative_simpson_nb(np.zeros(3, np.float64), 1.0)

class Solver:
    def solve(self, problem, **kwargs):
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        # compute cumulative Simpson and slice off the initial zero
        full = cumulative_simpson_nb(y, dx)
        return full[1:]