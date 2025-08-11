import numpy as np
import numba

@numba.njit(parallel=True, fastmath=True)
def _cumulative_simpson_numba(y_flat, dx):
    """
    Compute cumulative integral along last axis for flattened 2D array y_flat
    using composite Simpson's rule with averaging for odd intervals.
    """
    n_signals, N = y_flat.shape
    out = np.empty_like(y_flat)
    for i in range(n_signals):
        row = y_flat[i]
        out_row = out[i]
        # initial point
        out_row[0] = 0.0
        if N > 1:
            # first interval by trapezoidal rule
            out_row[1] = dx * 0.5 * (row[0] + row[1])
        for j in range(2, N):
            if (j & 1) != 0:
                # odd index: trapezoidal rule
                out_row[j] = out_row[j-1] + dx * 0.5 * (row[j-1] + row[j])
            else:
                # even index: Simpson rule on last two segments
                out_row[j] = out_row[j-2] + dx/3.0 * (row[j-2] + 4.0*row[j-1] + row[j])
    return out

class Solver:
    def __init__(self):
        # compile numba function at init (not counted in runtime)
        _ = _cumulative_simpson_numba(np.zeros((1, 3), dtype=np.float64), 1.0)

    def solve(self, problem, **kwargs):
        y2 = problem["y2"]
        dx = problem["dx"]
        from scipy.integrate import cumulative_simpson
        return cumulative_simpson(y2, dx=dx)