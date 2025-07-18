import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_lapack cimport dgesdd

def compute_svd(cnp.ndarray[cnp.float64_t, ndim=2] A):
    cdef:
        int m = A.shape[0]
        int n = A.shape[1]
        int k = min(m, n)
        char jobz = 'S'
        cnp.ndarray[cnp.float64_t, ndim=2] A_copy = A.copy()
        cnp.ndarray[cnp.float64_t, ndim=2] U = np.empty((m, k), dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=1] S = np.empty(k, dtype=np.float64)
        cnp.ndarray[cnp.float64_t, ndim=2] Vt = np.empty((k, n), dtype=np.float64)
        int lda = m
        int ldu = m
        int ldvt = k
        int lwork = -1
        double work_test
        int info = 0
        cnp.ndarray[double] work
        cnp.ndarray[int] iwork = np.empty(8*k, dtype=np.int32)
    
    # Query workspace size
    dgesdd(&jobz, &m, &n, &A_copy[0,0], &lda, &S[0], 
            &U[0,0], &ldu, &Vt[0,0], &ldvt, &work_test, &lwork, &iwork[0], &info)
    
    if info != 0:
        raise RuntimeError(f"Workspace query failed with error {info}")
    
    # Allocate workspace
    lwork = int(work_test)
    work = np.empty(lwork, dtype=np.float64)
    
    # Compute SVD
    dgesdd(&jobz, &m, &n, &A_copy[0,0], &lda, &S[0], 
            &U[0,0], &ldu, &Vt[0,0], &ldvt, &work[0], &lwork, &iwork[0], &info)
    
    if info != 0:
        raise RuntimeError(f"SVD computation failed with error {info}")
    
    return U, S, Vt