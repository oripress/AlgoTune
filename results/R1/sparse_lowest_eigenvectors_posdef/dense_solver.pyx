import numpy as np
cimport numpy as np
cimport cython
from scipy.linalg.cython_lapack cimport dsyevd

@cython.boundscheck(False)
@cython.wraparound(False)
def dense_eigvalsh(double[:, ::1] matrix):
    cdef int n = matrix.shape[0]
    cdef char jobz = 'N'  # Only eigenvalues
    cdef char uplo = 'L'  # Lower triangular
    cdef int info = 0
    cdef int lwork = -1
    cdef int liwork = -1
    cdef double work_query
    cdef int iwork_query
    
    # Allocate arrays
    cdef double[::1] w = np.empty(n, dtype=np.float64)
    cdef double[::1] a = matrix.copy().reshape(n*n)  # Copy because dsyevd overwrites
    cdef double[::1] work = np.empty(1, dtype=np.float64)
    cdef int[::1] iwork = np.empty(1, dtype=np.int32)
    
    # Query for optimal workspace
    dsyevd(&jobz, &uplo, &n, &a[0], &n, &w[0], &work_query, &lwork, &iwork_query, &liwork, &info)
    if info != 0:
        raise ValueError("LAPACK query failed")
    
    # Allocate workspace
    lwork = int(work_query)
    liwork = iwork_query
    work = np.empty(lwork, dtype=np.float64)
    iwork = np.empty(liwork, dtype=np.int32)
    
    # Compute eigenvalues
    dsyevd(&jobz, &uplo, &n, &a[0], &n, &w[0], &work[0], &lwork, &iwork[0], &liwork, &info)
    if info != 0:
        raise ValueError("LAPACK failed")
    
    return np.asarray(w)