# distutils: language = c
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def outer(cnp.ndarray[cnp.double_t, ndim=1] x not None,
          cnp.ndarray[cnp.double_t, ndim=1] y not None,
          cnp.ndarray[cnp.double_t, ndim=2] out not None):
    cdef int i, j, n = x.shape[0]
    cdef double xi
    with nogil:
        for i in prange(n, nogil=True):
            xi = x[i]
            for j in range(n):
                out[i, j] = xi * y[j]
# distutils: language = c
import numpy as np
cimport numpy as np

def outer(np.ndarray[np.double_t, ndim=1] a, np.ndarray[np.double_t, ndim=1] b):
    """
    Compute outer product a[:,None] * b with explicit C loops.
    """
    cdef int n = a.shape[0]
    # Ensure contiguous and correct dtype
    cdef np.ndarray[np.double_t, ndim=1] ca = np.ascontiguousarray(a, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] cb = np.ascontiguousarray(b, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=2] out = np.empty((n, n), dtype=np.double)
    cdef int i, j
    cdef double ai
    for i in range(n):
        ai = ca[i]
        for j in range(n):
            out[i, j] = ai * cb[j]
    return out