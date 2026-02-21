# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np

def compute_cumulative_simpson(double[:] y, double dx):
    cdef int n = y.shape[0]
    if n < 2:
        return np.empty(0, dtype=np.float64)
        
    cdef np.ndarray[np.float64_t, ndim=1] result = np.empty(n - 1, dtype=np.float64)
    
    if n < 3:
        result[0] = (y[0] + y[1]) / 2.0 * dx
        return result
        
    cdef double c1 = (dx / 3.0) * 1.25
    cdef double c2 = (dx / 3.0) * 2.0
    cdef double c3 = (dx / 3.0) * -0.25
    
    cdef int i
    cdef double current_sum = 0.0
    cdef double sub_integral
    
    for i in range(n - 1):
        if i % 2 == 0:
            # h1 interval
            sub_integral = c1 * y[i] + c2 * y[i+1] + c3 * y[i+2]
        else:
            # h2 interval
            sub_integral = c1 * y[i+1] + c2 * y[i] + c3 * y[i-1]
        
        current_sum += sub_integral
        result[i] = current_sum
        
    return result