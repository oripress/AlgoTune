# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cython.parallel import prange

def reconstruct_cython(double[:, ::1] vecs, double[::1] vals):
    cdef int n = vecs.shape[0]
    cdef int k = vecs.shape[1]
    cdef int i, j, l
    cdef double s
    cdef double[:, ::1] X = np.zeros((n, n), dtype=np.float64)

    # We can parallelize the outer loop
    # Note: prange requires OpenMP which might not be available.
    # If not available, it runs sequentially.
    
    for i in range(n):
        for j in range(i, n):
            s = 0.0
            for l in range(k):
                s += vecs[i, l] * vals[l] * vecs[j, l]
            X[i, j] = s
            X[j, i] = s
            
    return np.asarray(X)