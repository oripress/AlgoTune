# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.math cimport fabs

def solve_cython(double[::1] v, double k):
    cdef int n = v.shape[0]
    cdef int i
    cdef double sum_abs = 0.0
    cdef double val
    
    # Compute sum of abs(v) to check if projection is needed
    # and to get initial guess
    for i in prange(n, nogil=True, num_threads=4):
        val = fabs(v[i])
        sum_abs += val
        
    if sum_abs <= k:
        return np.asarray(v)
        
    # Newton's method to find theta
    # f(theta) = sum(max(|v_i| - theta, 0)) - k = 0
    # theta_new = (sum_{|v_i|>theta} |v_i| - k) / count_{|v_i|>theta}
    
    cdef double theta = (sum_abs - k) / n
    cdef double theta_new
    cdef double s_active
    cdef int n_active
    cdef int iter_count = 0
    cdef int max_iter = 20 # Should converge very fast
    
    while iter_count < max_iter:
        s_active = 0.0
        n_active = 0
        
        for i in prange(n, nogil=True, num_threads=4):
            val = fabs(v[i])
            if val > theta:
                s_active += val
                n_active += 1
        
        if n_active == 0:
            # Should not happen if k < sum_abs
            theta = 0.0
            break
            
        theta_new = (s_active - k) / n_active
        
        # Check convergence
        if fabs(theta_new - theta) < 1e-9:
            theta = theta_new
            break
            
        theta = theta_new
        iter_count += 1
        
    # Compute w
    cdef double[::1] w = np.empty(n, dtype=np.float64)
    cdef double diff
    
    for i in prange(n, nogil=True, num_threads=4):
        if v[i] > 0:
            diff = v[i] - theta
            if diff > 0:
                w[i] = diff
            else:
                w[i] = 0.0
        elif v[i] < 0:
            if -v[i] > theta:
                w[i] = v[i] + theta
            else:
                w[i] = 0.0
        else:
            w[i] = 0.0
            
    return np.asarray(w)