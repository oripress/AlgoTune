import numpy as np
import math
from numba import njit, prange
@njit(cache=True)
def eval_poly(coeffs, x):
    res = coeffs[0]
    for k in range(1, coeffs.shape[0]):
        res = res * x + coeffs[k]
    return res

@njit(cache=True, parallel=True, fastmath=True)
def aberth(coeffs, deriv, tol, max_iter):
    n = coeffs.shape[0] - 1
    # Cauchy bound for roots
    a0 = coeffs[0] if coeffs[0] != 0.0 else 1.0
    max_ratio = 0.0
    for i in range(1, coeffs.shape[0]):
        r = abs(coeffs[i]) / a0
        if r > max_ratio:
            max_ratio = r
    R = 1.0 + max_ratio

    # initial guesses on circle |z|=R
    x = np.empty(n, dtype=np.complex128)
    for i in range(n):
        ang = 2 * math.pi * i / n
        x[i] = R * (math.cos(ang) + 1j * math.sin(ang))

    # iterative Aberth refinement
    for _ in range(max_iter):
        converged = True
        for i in range(n):
            xi = x[i]
            # evaluate p and p'
            pv = coeffs[0]
            for j in range(1, coeffs.shape[0]):
                pv = pv * xi + coeffs[j]
            dpv = deriv[0]
            for j in range(1, deriv.shape[0]):
                dpv = dpv * xi + deriv[j]
            # compute Aberth correction sum
            sum_term = 0+0j
            for j in range(n):
                if j != i:
                    sum_term += 1.0 / (xi - x[j])
            denom = dpv - pv * sum_term
            delta = pv / denom
            if abs(delta) > tol:
                converged = False
            x[i] = xi - delta
        if converged:
            break
    return x

# warm up compilation
_dummy_c = np.array([1.0, -1.0], dtype=np.float64)
_dummy_d = np.array([1.0], dtype=np.float64)
_ = aberth(_dummy_c, _dummy_d, 1e-3, 1)

class Solver:
    def solve(self, problem, **kwargs):
        coeffs = np.array(problem, dtype=np.float64)
        deg = coeffs.size - 1
        # small-degree fallback
        if deg <= 60:
            roots = np.roots(coeffs)
            roots = np.real_if_close(roots, tol=1e-3)
            roots = np.real(roots)
            roots.sort()
            return roots[::-1].tolist()
        # build derivative
        deriv = coeffs[:-1] * np.arange(deg, 0, -1, dtype=np.float64)
        # Aberth method
        roots = aberth(coeffs, deriv, 1e-3, 40)
        # all roots are real
        roots = np.real(roots)
        roots.sort()
        return roots[::-1].tolist()