import numpy as np
cimport numpy as np
from scipy.linalg.cython_lapack cimport dsyev

def compute_eigenvalues(double[:, :] matrix):
    cdef int n = matrix.shape[0]
    cdef int lda = n
    cdef int info
    cdef char jobz = 'N'  # Compute eigenvalues only
    cdef char uplo = 'L'  # Lower triangular matrix
    
    # Allocate memory for eigenvalues
    cdef double[:] eigenvalues = np.empty(n, dtype=np.float64)
    
    # Allocate workspace
    cdef double work_query
    cdef int lwork = -1
    dsyev(&jobz, &uplo, &n, &matrix[0,0], &lda, &eigenvalues[0], &work_query, &lwork, &info)
    
    # Determine optimal workspace size
    lwork = int(work_query)
    cdef double[:] work = np.empty(lwork, dtype=np.float64)
    
    # Compute eigenvalues
    dsyev(&jobz, &uplo, &n, &matrix[0,0], &lda, &eigenvalues[0], &work[0], &lwork, &info)
    
    if info != 0:
        raise RuntimeError(f"LAPACK dsyev failed with error code {info}")
    
    return np.asarray(eigenvalues)