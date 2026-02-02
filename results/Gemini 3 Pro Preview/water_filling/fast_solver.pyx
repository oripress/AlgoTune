# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport log, abs
from libc.string cimport memcpy

def solve_cython(list alpha_list, double P_total):
    cdef int n = len(alpha_list)
    if n == 0 or P_total <= 0:
        return np.full(n, np.nan), np.nan

    # Allocate memory for alpha
    cdef double* alpha = <double*>malloc(n * sizeof(double))
    
    if not alpha:
        raise MemoryError()

    cdef int i
    cdef double val
    cdef double current_sum = 0.0
    cdef bint valid = True
    
    # Parse list to C array
    for i in range(n):
        val = alpha_list[i]
        if val <= 0:
            valid = False
            break
        alpha[i] = val
        current_sum += val
        
    if not valid:
        free(alpha)
        return np.full(n, np.nan), np.nan

    # Iterative Water Filling
    cdef double* alpha_work = <double*>malloc(n * sizeof(double))
    if not alpha_work:
        free(alpha)
        raise MemoryError()
        
    memcpy(alpha_work, alpha, n * sizeof(double))
        
    cdef int k = n
    cdef double w = (P_total + current_sum) / k
    cdef double removed_sum
    cdef int removed_count
    
    while True:
        removed_count = 0
        removed_sum = 0.0
        i = 0
        while i < k:
            if alpha_work[i] >= w:
                val = alpha_work[i]
                removed_sum += val
                removed_count += 1
                k -= 1
                alpha_work[i] = alpha_work[k]
            else:
                i += 1
        
        if removed_count == 0:
            break
            
        current_sum -= removed_sum
        w = (P_total + current_sum) / k
        
    # Compute x and capacity
    cdef double capacity = 0.0
    cdef double total_power = 0.0
    cdef double log_w = log(w)
    
    cdef np.ndarray[np.float64_t, ndim=1] x_arr = np.empty(n, dtype=np.float64)
    cdef double[:] x_view = x_arr
    
    for i in range(n):
        val = w - alpha[i]
        if val > 0:
            x_view[i] = val
            capacity += log_w
            total_power += val
        else:
            x_view[i] = 0.0
            capacity += log(alpha[i])
            
    # Rescale
    cdef double scaling
    if total_power > 1e-12:
        scaling = P_total / total_power
        if abs(scaling - 1.0) > 1e-9:
            total_power = 0.0
            capacity = 0.0
            for i in range(n):
                x_view[i] *= scaling
                capacity += log(alpha[i] + x_view[i])
                
    free(alpha)
    free(alpha_work)
    
    return x_arr, capacity