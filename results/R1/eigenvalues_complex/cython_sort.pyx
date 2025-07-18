import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_sort(np.ndarray[np.complex128_t, ndim=1] eigenvalues):
    cdef int n = eigenvalues.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] real = np.ascontiguousarray(-np.real(eigenvalues))
    cdef np.ndarray[np.float64_t, ndim=1] imag = np.ascontiguousarray(-np.imag(eigenvalues))
    
    # Create keys array
    cdef np.ndarray[np.float64_t, ndim=2] keys = np.empty((n, 2), dtype=np.float64)
    cdef int i
    for i in range(n):
        keys[i, 0] = real[i]
        keys[i, 1] = imag[i]
    
    # Get sorted indices using lexsort
    cdef np.ndarray idx = np.lexsort((keys[:, 1], keys[:, 0]))
    return idx