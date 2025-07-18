#cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, fabs, M_PI

def durand_kerner(np.ndarray[double, ndim=1] coeffs):
    """
    Durand-Kerner root-finding for polynomials with all real roots.
    """
    cdef int n = coeffs.shape[0] - 1
    if n <= 0:
        # No roots for constant polynomial
        return np.empty(0, dtype=np.complex128)
    cdef double An = coeffs[0]
    # Find radius R = 1 + max |a_i/a_n|
    cdef double maxAi = 0.0, val, R
    cdef int i, j, iters, max_iter = 50
    cdef double tol = 1e-6, max_delta
    cdef np.ndarray[np.complex128_t, ndim=1] roots = np.empty(n, dtype=np.complex128)
    cdef complex p, prod, delta
    # Compute bounding circle radius
    for i in range(1, n+1):
        val = fabs(coeffs[i] / An)
        if val > maxAi:
            maxAi = val
    R = 1.0 + maxAi
    # Initialize guesses on circle
    for i in range(n):
        angle = 2.0 * M_PI * i / n
        roots[i] = R * (cos(angle) + sin(angle) * 1j)
    # Durand-Kerner iterations
    for iters in range(max_iter):
        max_delta = 0.0
        for i in range(n):
            # Evaluate polynomial at roots[i] via Horner
            p = coeffs[0]
            for j in range(1, n+1):
                p = p * roots[i] + coeffs[j]
            # Compute product of differences
            prod = 1.0 + 0.0j
            for j in range(n):
                if j != i:
                    prod *= (roots[i] - roots[j])
            # Update root
            delta = p / prod
            roots[i] -= delta
            # Track max correction
            # Use absolute sum for performance
            val = fabs(delta.real) + fabs(delta.imag)
            if val > max_delta:
                max_delta = val
        if max_delta < tol:
            break
    return roots