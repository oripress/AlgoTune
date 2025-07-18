import numpy as np
cimport numpy as np
cimport cython
from scipy.linalg.cython_lapack cimport dsyev, ssyev

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_normalize_affinity(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=1] D_inv_sqrt):
    """Cython-optimized affinity matrix normalization."""
    cdef int n = A.shape[0]
    cdef int i, j
    cdef float di, dj
    
    for i in range(n):
        di = D_inv_sqrt[i]
        for j in range(n):
            dj = D_inv_sqrt[j]
            A[i, j] = A[i, j] * di * dj
    return A

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_eigh(np.ndarray[np.float32_t, ndim=2] A, int n_clusters):
    """Cython-optimized eigen decomposition for small matrices."""
    cdef int n = A.shape[0]
    cdef char jobz = 'V'  # Compute eigenvectors
    cdef char uplo = 'L'  # Lower triangular
    cdef int lda = n
    cdef int lwork = 3*n - 1
    cdef int info = 0
    
    # Copy matrix since it will be overwritten
    cdef np.ndarray[np.float32_t, ndim=2] A_copy = A.copy()
    
    # Workspace arrays
    cdef np.ndarray[np.float32_t, ndim=1] work = np.zeros(lwork, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] w = np.zeros(n, dtype=np.float32)
    
    # Call LAPACK function
    ssyev(&jobz, &uplo, &n, <float*>A_copy.data, &n, <float*>w.data, <float*>work.data, &lwork, &info)
    
    if info != 0:
        raise ValueError("Eigen decomposition failed")
    
    # Get largest eigenvectors
    cdef np.ndarray[np.int64_t, ndim=1] idx = np.argsort(w)[::-1][:n_clusters]
    cdef np.ndarray[np.float32_t, ndim=2] eigenvectors = A_copy[:, idx]
    
    return w[idx], eigenvectors