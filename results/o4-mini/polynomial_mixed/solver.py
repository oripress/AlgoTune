import numpy as np
import math, cmath
from numba import njit

@njit(fastmath=True, cache=True)
def durand_kerner(coeffs):
    # Weierstrass-Durand-Kerner method
    n = coeffs.shape[0] - 1
    a0 = coeffs[0]
    # estimate root radius
    m = 0.0
    for i in range(1, n+1):
        v = abs(coeffs[i] / a0)
        if v > m:
            m = v
    R = 1.0 + m
    # initial guesses on a circle
    z = np.empty(n, dtype=np.complex128)
    for k in range(n):
        angle = 2 * math.pi * k / n
        z[k] = R * complex(math.cos(angle), math.sin(angle))
    tol = 1e-10
    max_iter = 60
    # root refinement
    for _ in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            xi = z[i]
            # Horner to eval p(xi)
            p = coeffs[0]
            for j in range(1, n+1):
                p = p * xi + coeffs[j]
            # denom = ∏_{j≠i} (xi - z[j])
            denom = 1.0 + 0.0j
            for j in range(n):
                if j != i:
                    denom *= (xi - z[j])
            delta = p / denom
            z[i] = xi - delta
            ad = abs(delta)
            if ad > max_delta:
                max_delta = ad
        if max_delta < tol:
            break
    return z

class Solver:
    def __init__(self):
        # force JIT compile on a tiny example
        _ = durand_kerner(np.array([1.0, -1.0], dtype=np.float64))

    def solve(self, problem, **kwargs):
        arr = np.array(problem, dtype=np.float64)
        n = arr.shape[0] - 1
        if n < 1:
            return np.array([], dtype=np.complex128)
        if n == 1:
            # trivial linear
            root = -arr[1] / arr[0]
            return np.array([complex(root)], dtype=np.complex128)
        if n == 2:
            # direct quadratic formula
            a, b, c = arr
            disc = b*b - 4*a*c
            sd = cmath.sqrt(disc)
            roots = np.array([(-b+sd)/(2*a), (-b-sd)/(2*a)], dtype=np.complex128)
            return -np.sort_complex(-roots)
        # general >2: Durand–Kerner
        roots = durand_kerner(arr)
        # sort descending by real, then imaginary
        return -np.sort_complex(-roots)