# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libcpp.algorithm cimport nth_element
from cython.operator cimport dereference as deref

def solve_cython(double[:] v, int k):
    cdef int n = v.shape[0]
    cdef int i
    cdef double val
    cdef double T
    
    # Create an array for absolute values
    # We use numpy array to manage memory easily
    cdef double[:] abs_v = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        val = v[i]
        if val < 0:
            abs_v[i] = -val
        else:
            abs_v[i] = val
            
    # Find the k-th largest element
    cdef int threshold_idx = n - k
    
    # Get pointer to data
    cdef double* abs_v_ptr = &abs_v[0]
    
    nth_element(abs_v_ptr, abs_v_ptr + threshold_idx, abs_v_ptr + n)
    
    T = abs_v_ptr[threshold_idx]
    
    # Count greater
    cdef int count_greater = 0
    cdef double abs_val
    
    for i in range(n):
        val = v[i]
        if val < 0:
            abs_val = -val
        else:
            abs_val = val
            
        if abs_val > T:
            count_greater += 1
            
    cdef int needed = k - count_greater
    cdef int equals_kept = 0
    
    # Apply pruning
    for i in range(n - 1, -1, -1):
        val = v[i]
        if val < 0:
            abs_val = -val
        else:
            abs_val = val
            
        if abs_val > T:
            # Keep
            pass
        elif abs_val == T:
            if equals_kept < needed:
                equals_kept += 1
                # Keep
            else:
                v[i] = 0.0
        else:
            v[i] = 0.0
            
    return v