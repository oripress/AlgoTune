# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dgesv

def solve_fast(object A_list, object b_list):
    cdef int n = len(b_list)
    cdef np.ndarray[double, ndim=2, mode='fortran'] A = np.array(A_list, dtype=np.float64, order='F')
    cdef np.ndarray[double, ndim=1] b = np.array(b_list, dtype=np.float64)
    
    cdef int nrhs = 1
    cdef int lda = n
    cdef int ldb = n
    cdef int info = 0
    cdef np.ndarray[int, ndim=1] ipiv = np.empty(n, dtype=np.int32)
    
    dgesv(&n, &nrhs, <double*> A.data, &lda, <int*> ipiv.data, <double*> b.data, &ldb, &info)
    
    return b.tolist()