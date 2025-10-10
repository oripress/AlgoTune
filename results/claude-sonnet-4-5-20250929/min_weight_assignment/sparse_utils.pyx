# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def build_dense_from_csr(double[:] data, int[:] indices, int[:] indptr, int n):
    """Build dense matrix from CSR format with inf for missing edges."""
    cdef int i, j, idx
    cdef double[:, :] dense_mat = np.full((n, n), np.inf, dtype=np.float64)
    
    for i in range(n):
        for idx in range(indptr[i], indptr[i + 1]):
            j = indices[idx]
            dense_mat[i, j] = data[idx]
    
    return np.asarray(dense_mat)