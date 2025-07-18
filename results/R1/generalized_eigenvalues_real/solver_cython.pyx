import numpy as np
cimport numpy as np
cimport cython
from scipy.linalg.cython_lapack cimport dsygvd

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_gep(double[:, ::1] A, double[:, ::1] B, int n):
    cdef int itype = 1  # A*x = lambda*B*x
    cdef char jobz = 'N'  # Compute eigenvalues only
    cdef char uplo = 'L'  # Lower triangular
    cdef int lda = n
    cdef int ldb = n
    cdef int lwork = max(1, 3*n-1)  # Minimum workspace size
    cdef double[::1] work = np.empty(lwork, dtype=np.float64)
    cdef int liwork = 1  # Not used for eigenvalues only
    cdef int[::1] iwork = np.empty(liwork, dtype=np.int32)
    cdef int info = 0
    cdef double[::1] eigenvalues = np.empty(n, dtype=np.float64)
    
    # Make copies since dsygvd overwrites input
    cdef double[:, ::1] A_copy = A.copy()
    cdef double[:, ::1] B_copy = B.copy()
    
    dsygvd(&itype, &jobz, &uplo, &n,
           &A_copy[0,0], &lda,
           &B_copy[0,0], &ldb,
           &eigenvalues[0],
           &work[0], &lwork,
           &iwork[0], &liwork, &info)
    
    if info != 0:
        raise ValueError("dsygvd failed with info = %d" % info)
    
    return np.asarray(eigenvalues)