import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cumulative_simpson_cy(double[::1] y, double dx):
    cdef int n = y.shape[0]
    cdef np.ndarray[double, ndim=1] result = np.zeros(n - 1, dtype=np.float64)
    cdef double[::1] result_view = result
    cdef int i
    
    if n < 2:
        return result
    
    result_view[0] = 0.5 * dx * (y[0] + y[1])
    
    if n < 3:
        return result
    
    result_view[1] = dx / 3.0 * (y[0] + 4.0 * y[1] + y[2])
    
    i = 2
    while i < n - 1:
        result_view[i] = result_view[i-1] + 0.5 * dx * (y[i] + y[i+1])
        i += 1
        if i < n - 1:
            result_view[i] = result_view[i-2] + dx / 3.0 * (y[i-1] + 4.0 * y[i] + y[i+1])
            i += 1
    
    return result