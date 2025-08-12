# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: infer_types=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
cimport cython

from cython cimport parallel
from libc.math cimport fabs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def cython_opt_power_control(double[:, ::1] G, double[::1] σ, double[::1] P_min, double[::1] P_max, double S_min, int n):
    """Ultra-optimized Cython power control with direct memory access"""
    cdef double[::1] P = np.empty(n, dtype=np.float64)
    cdef double interf, P_i, G_ii, P_min_i, P_max_i
    cdef int i, j
    cdef double[::1] G_i
    
    # Parallel execution with optimal CPU utilization
    for i in cython.parallel.prange(n, nogil=True, schedule='static'):
        interf = σ[i]
        G_i = G[i]
        G_ii = G_i[i]
        P_min_i = P_min[i]
        P_max_i = P_max[i]
        
        # Ultra-optimized computation with direct memory access
        for j in range(n):
            if j != i:
                interf += G_i[j] * P_min[j]
        P_i = S_min * interf / G_ii
        
        # Ultra-optimized bounds check
        if P_i <= P_min_i:
            P[i] = P_min_i
        elif P_i >= P_max_i:
            P[i] = P_max_i
        else:
            P[i] = P_i
    
    return np.asarray(P)