# distutils: language=c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dgesv

def cython_solve(double[:, :] A, double[:] b):
    cdef int n = A.shape[0]
    cdef int nrhs = 1
    cdef int lda = n
    cdef int ldb = n
    cdef int info
    cdef int[:] ipiv = np.zeros(n, dtype=np.int32)
    
    # Make copies since dgesv overwrites inputs
    cdef double[:, :] A_copy = A.copy()
    cdef double[:] b_copy = b.copy()
    
    dgesv(&n, &nrhs, &A_copy[0,0], &lda, &ipiv[0], &b_copy[0], &ldb, &info)
    
    if info != 0:
        raise ValueError("Matrix is singular")
    
    return np.asarray(b_copy)