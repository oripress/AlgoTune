# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dgges

def compute_qz(np.ndarray A_in, np.ndarray B_in):
    cdef int n = A_in.shape[0]
    
    cdef np.ndarray[double, ndim=2, mode="fortran"] A = np.array(A_in, dtype=np.float64, order='F', copy=True)
    cdef np.ndarray[double, ndim=2, mode="fortran"] B = np.array(B_in, dtype=np.float64, order='F', copy=True)
    
    cdef np.ndarray[double, ndim=2, mode="fortran"] vsl = np.empty((n, n), dtype=np.float64, order='F')
    cdef np.ndarray[double, ndim=2, mode="fortran"] vsr = np.empty((n, n), dtype=np.float64, order='F')
    
    cdef np.ndarray[double, ndim=1] alphar = np.empty(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] alphai = np.empty(n, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] beta = np.empty(n, dtype=np.float64)
    
    cdef int sdim = 0
    cdef int info = 0
    
    cdef int lwork = 8 * n + 16
    if lwork < 1:
        lwork = 1
    cdef np.ndarray[double, ndim=1] work = np.empty(lwork, dtype=np.float64)
    
    cdef char jobvsl = b'V'
    cdef char jobvsr = b'V'
    cdef char sort = b'N'
    
    dgges(&jobvsl, &jobvsr, &sort, NULL, &n, &A[0,0], &n, &B[0,0], &n, &sdim, &alphar[0], &alphai[0], &beta[0], &vsl[0,0], &n, &vsr[0,0], &n, &work[0], &lwork, NULL, &info)
    
    return A, B, vsl, vsr