import numpy as np
import numba as nb

@nb.njit(nogil=True, fastmath=True)
def cum_simp_nb(y, dx):
    n = y.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float64)
    n_out = n - 1
    out = np.empty(n_out, dtype=np.float64)
    half_dx = 0.5 * dx
    third_dx = dx / 3.0
    # first trapezoid
    out[0] = half_dx * (y[0] + y[1])
    sum_odd = y[1]
    sum_even = 0.0
    for i in range(2, n, 2):
        # Simpson at even i
        out[i - 1] = third_dx * (y[0] + y[i] + 4.0 * sum_odd + 2.0 * sum_even)
        sum_even += y[i]
        # trapezoid at odd i+1
        if i + 1 < n:
            sum_odd += y[i + 1]
            out[i] = out[i - 1] + half_dx * (y[i] + y[i + 1])
    return out

# warm-up compile to avoid JIT during solve
_cum_simp_nb_dummy = cum_simp_nb(np.zeros(2, dtype=np.float64), 1.0)

class Solver:
    def solve(self, problem, **kwargs):
        y = np.asarray(problem["y"], dtype=np.float64)
        dx = float(problem["dx"])
        return cum_simp_nb(y, dx)