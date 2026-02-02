import numpy as np
cimport numpy as np
cimport cython
from scipy import signal

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.double_t, ndim=1] direct_convolve(double[:] a, double[:] b, int sa, int sb):
    cdef int out_len = sa + sb - 1
    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(out_len, dtype=np.double)
    cdef double[:] out_view = out
    cdef int i, j
    cdef double val_a
    
    for i in range(sa):
        val_a = a[i]
        for j in range(sb):
            out_view[i + j] += val_a * b[j]
            
    return out

def solve_cython(list problem):
    cdef list results = []
    cdef np.ndarray[np.double_t, ndim=1] a_arr, b_arr
    cdef double[:] a_view, b_view
    cdef int sa, sb, target_len
    
    for p in problem:
        a_in, b_in = p
        # Ensure inputs are numpy arrays of double
        # Using np.array(..., copy=False) might be faster if already array
        a_arr = np.array(a_in, dtype=np.double, copy=False)
        b_arr = np.array(b_in, dtype=np.double, copy=False)
        
        # Ensure contiguous for memoryview efficiency
        if not a_arr.flags['C_CONTIGUOUS']:
            a_arr = np.ascontiguousarray(a_arr)
        if not b_arr.flags['C_CONTIGUOUS']:
            b_arr = np.ascontiguousarray(b_arr)
            
        sa = a_arr.shape[0]
        sb = b_arr.shape[0]
        target_len = sa + sb - 1
        
        # Threshold for direct convolution
        # Lower threshold to catch more small cases
        # Also, ensure we are using the most efficient loop order
        # If sa > sb, swap for better cache locality in inner loop?
        # Actually, inner loop iterates over b. If b is small, it fits in cache.
        # If b is large, we might have cache misses.
        # But here we are targeting small arrays.
        
        if sa * sb < 250000: 
            a_view = a_arr
            b_view = b_arr
            results.append(direct_convolve(a_view, b_view, sa, sb))
        else:
            results.append(signal.convolve(a_arr, b_arr, mode='full'))
            
    return results