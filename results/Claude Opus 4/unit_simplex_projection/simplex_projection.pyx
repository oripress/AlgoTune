import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def project_simplex_cython(np.ndarray[np.float64_t, ndim=1] y):
    """
    Cython-optimized projection onto the probability simplex.
    """
    cdef int n = y.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] sorted_y = np.sort(y)[::-1]
    cdef np.ndarray[np.float64_t, ndim=1] x = np.zeros(n, dtype=np.float64)
    
    cdef double cumsum = 0.0
    cdef int rho = 0
    cdef double temp
    cdef int i
    
    # Find rho
    for i in range(n):
        cumsum += sorted_y[i]
        temp = (cumsum - 1.0) / (i + 1)
        if sorted_y[i] > temp:
            rho = i
    
    # Compute theta
    cumsum = 0.0
    for i in range(rho + 1):
        cumsum += sorted_y[i]
    cdef double theta = (cumsum - 1.0) / (rho + 1)
    
    # Project onto the simplex
    for i in range(n):
        x[i] = max(y[i] - theta, 0.0)
    
    return x