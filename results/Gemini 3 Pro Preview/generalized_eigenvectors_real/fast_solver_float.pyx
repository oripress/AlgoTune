# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport ssygvd
from libc.stdlib cimport malloc, free

def solve_cython_float(float[::1, :] A, float[::1, :] B):
    cdef int n = A.shape[0]
    cdef int itype = 1
    cdef char jobz = 86  # 'V'
    cdef char uplo = 76  # 'L'
    cdef int lda = n
    cdef int ldb = n
    cdef int info = 0
    
    # Workspace query
    # lwork >= 1 + 6*N + 2*N**2
    cdef int lwork = 1 + 6*n + 2*n*n
    cdef int liwork = 3 + 5*n
    
    cdef float[::1] w = np.empty(n, dtype=np.float32)
    
    cdef float *work = <float *>malloc(lwork * sizeof(float))
    cdef int *iwork = <int *>malloc(liwork * sizeof(int))
    
    if work == NULL or iwork == NULL:
        if work != NULL: free(work)
        if iwork != NULL: free(iwork)
        raise MemoryError()
        
    try:
        ssygvd(&itype, &jobz, &uplo, &n, &A[0,0], &lda, &B[0,0], &ldb, &w[0], work, &lwork, iwork, &liwork, &info)
    finally:
        free(work)
        free(iwork)
        
    # Construct output lists in descending order
    cdef list w_list = [None] * n
    cdef list v_list = [None] * n
    cdef list vec
    cdef int i, j
    
    for i in range(n):
        w_list[n - 1 - i] = w[i]
        vec = [None] * n
        for j in range(n):
            vec[j] = A[j, i]
        v_list[n - 1 - i] = vec
        
    return w_list, v_list