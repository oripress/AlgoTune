import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def list_to_array(list matrix, int n, int m):
    # Create float64 array
    cdef np.ndarray[np.float64_t, ndim=2] arr = np.empty((n, m), dtype=np.float64)
    cdef int i, j
    cdef list row
    cdef double val
    
    for i in range(n):
        row = matrix[i]
        for j in range(m):
            val = row[j]
            arr[i, j] = val
            
    return arr