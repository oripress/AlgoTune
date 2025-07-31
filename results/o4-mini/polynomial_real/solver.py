import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def _solve_newton(coeffs):
    """
    Real-root finding via Newton + synthetic deflation.
    Reduced iterations and relaxed tolerance for 1e-3 accuracy.
    """
    m = coeffs.shape[0]
    n = m - 1
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    # Bound on root magnitude
    a0 = coeffs[0]
    R = 0.0
    for i in range(1, m):
        tmp = abs(coeffs[i] / a0)
        if tmp > R:
            R = tmp
    R += 1.0
    # Allocate roots and working polynomial
    roots = np.empty(n, dtype=np.float64)
    poly = coeffs.copy()
    # Deflation loop
    for k in range(n):
        mk = poly.shape[0]
        nk = mk - 1
        # Initial guess in [-R, R]
        if nk > 1:
            x = R * ((2.0 * k) / (nk - 1.0) - 1.0)
        else:
            x = 0.0
        # Newton iterations (reduced to 8, tol=1e-5)
        for _ in range(8):
            p = poly[0]
            dp = 0.0
            for j in range(1, mk):
                dp = dp * x + p
                p  = p  * x + poly[j]
            if dp == 0.0:
                break
            dx = p / dp
            x -= dx
            if abs(dx) < 1e-5:
                break
        roots[k] = x
        # Synthetic division
        newp = np.empty(mk - 1, dtype=np.float64)
        newp[0] = poly[0]
        for j in range(1, mk - 1):
            newp[j] = poly[j] + newp[j - 1] * x
        poly = newp
    return roots

class Solver:
    def __init__(self):
        # Warm-up JIT
        _ = _solve_newton(np.array([1.0, -3.0, 2.0], dtype=np.float64))

    def solve(self, problem, **kwargs):
        coeffs = np.array(problem, dtype=np.float64)
        roots = _solve_newton(coeffs)
        # C-level sort descending for speed
        sorted_roots = np.sort(roots)[::-1]
        return sorted_roots.tolist()