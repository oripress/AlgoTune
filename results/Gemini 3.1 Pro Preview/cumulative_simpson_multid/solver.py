import numpy as np
from scipy.integrate import cumulative_simpson

import numba
from numba import njit

@njit(parallel=True, fastmath=True, boundscheck=False, nogil=True)
def _compute_cumulative_simpson(y, dx):
    shape = y.shape
    N = shape[-1]
    y_flat = y.reshape((-1, N))
    M = y_flat.shape[0]
    
    out = np.empty((M, N - 1), dtype=y.dtype)
    
    c1 = dx * (1.25 / 3.0)
    c2 = dx * (2.0 / 3.0)
    c3 = dx * (-0.25 / 3.0)
    
    for i in numba.prange(M):  # pylint: disable=not-an-iterable
        acc = 0.0
        y0 = y_flat[i, 0]
        y1 = y_flat[i, 1]
        j = 0
        while j < N - 3:
            y2 = y_flat[i, j+2]
            c2y1 = c2 * y1
            
            val0 = c1 * y0 + c2y1 + c3 * y2
            acc += val0
            out[i, j] = acc
            
            val1 = c1 * y2 + c2y1 + c3 * y0
            acc += val1
            out[i, j+1] = acc
            
            y0 = y2
            y1 = y_flat[i, j+3]
            j += 2
            
        if j < N - 2:
            y2 = y_flat[i, j+2]
            c2y1 = c2 * y1
            
            val0 = c1 * y0 + c2y1 + c3 * y2
            acc += val0
            out[i, j] = acc
            
            val1 = c1 * y2 + c2y1 + c3 * y0
            acc += val1
            out[i, j+1] = acc
            j += 2
            
        if N % 2 == 0:
            val = c1 * y_flat[i, N-1] + c2 * y_flat[i, N-2] + c3 * y_flat[i, N-3]
            acc += val
            out[i, N-2] = acc
            
    return out.reshape(shape[:-1] + (N - 1,))

class Solver:
    def solve(self, problem: dict, **kwargs):
        return _compute_cumulative_simpson(problem["y2"], problem["dx"])