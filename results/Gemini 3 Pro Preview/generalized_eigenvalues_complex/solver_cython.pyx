import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport sggev

def solve_fast(float[::1, :] A, float[::1, :] B):
    cdef int n = A.shape[0]
    cdef int lda = n
    cdef int ldb = n
    cdef int ldvl = 1
    cdef int ldvr = 1
    cdef int info = 0
    
    # Arrays for results
    cdef float[:] alphar = np.empty(n, dtype=np.float32)
    cdef float[:] alphai = np.empty(n, dtype=np.float32)
    cdef float[:] beta = np.empty(n, dtype=np.float32)
    
    cdef float dummy = 0
    
    # Workspace query
    cdef float work_query = 0
    cdef int lwork_query = -1
    
    sggev('N', 'N', &n, &A[0,0], &lda, &B[0,0], &ldb, 
          &alphar[0], &alphai[0], &beta[0], 
          &dummy, &ldvl, &dummy, &ldvr, 
          &work_query, &lwork_query, &info)
          
    cdef int optimal_lwork = <int>work_query
    cdef float[:] work = np.empty(optimal_lwork, dtype=np.float32)
    
    sggev('N', 'N', &n, &A[0,0], &lda, &B[0,0], &ldb, 
          &alphar[0], &alphai[0], &beta[0], 
          &dummy, &ldvl, &dummy, &ldvr, 
          &work[0], &optimal_lwork, &info)
          
    # Compute complex eigenvalues
    cdef float complex[:] vals = np.empty(n, dtype=np.complex64)
    cdef int i
    cdef float b
    
    for i in range(n):
        b = beta[i]
        vals[i] = (alphar[i] / b) + 1j * (alphai[i] / b)
        
    return np.asarray(vals)