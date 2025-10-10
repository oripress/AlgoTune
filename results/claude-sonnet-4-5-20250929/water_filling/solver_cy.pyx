# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp
from libc.math cimport log, fabs
cimport cython

ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple water_filling_core(cnp.ndarray[DTYPE_t, ndim=1] alpha, double P_total):
    """Optimized Cython water-filling using Newton's method."""
    cdef int n = alpha.shape[0]
    cdef int i, iter, count
    cdef double w, f_w, df_w, x_sum, scaling
    cdef cnp.ndarray[DTYPE_t, ndim=1] x_opt = np.empty(n, dtype=np.float64)
    cdef double capacity = 0.0
    cdef double* alpha_ptr = <double*>alpha.data
    cdef double* x_ptr = <double*>x_opt.data
    cdef double alpha_sum = 0.0
    
    # Initial guess: assume all channels active
    for i in range(n):
        alpha_sum += alpha_ptr[i]
    w = (P_total + alpha_sum) / n
    
    # Newton's method: find w such that sum(max(0, w - alpha_i)) = P_total
    for iter in range(15):  # Newton converges much faster
        f_w = 0.0
        count = 0
        for i in range(n):
            if w > alpha_ptr[i]:
                f_w += w - alpha_ptr[i]
                count += 1
        f_w -= P_total
        
        if fabs(f_w) < 1e-12:
            break
        
        if count == 0:
            count = 1  # Avoid division by zero
        
        df_w = <double>count
        w -= f_w / df_w
    
    # Compute allocation
    x_sum = 0.0
    for i in range(n):
        if w > alpha_ptr[i]:
            x_ptr[i] = w - alpha_ptr[i]
            x_sum += x_ptr[i]
        else:
            x_ptr[i] = 0.0
    
    # Rescale to match exact budget
    if x_sum > 0:
        scaling = P_total / x_sum
        for i in range(n):
            x_ptr[i] *= scaling
    
    # Compute capacity
    for i in range(n):
        capacity += log(alpha_ptr[i] + x_ptr[i])
    
    return x_opt, capacity