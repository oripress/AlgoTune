import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_sort_roots(np.ndarray[np.complex128_t, ndim=1] roots):
    """Cython-accelerated root sorting with vectorized operations."""
    cdef int n = roots.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] real_parts = np.real(roots)
    cdef np.ndarray[np.float64_t, ndim=1] imag_parts = np.imag(roots)
    
    # Create an array of indices
    cdef np.ndarray[np.int_t, ndim=1] indices = np.arange(n, dtype=np.int_)
    
    # Sort indices based on real parts (descending) and then imaginary parts (descending)
    indices = indices[np.lexsort((-imag_parts, -real_parts))]
    
    # Return sorted roots
    return roots[indices].tolist()