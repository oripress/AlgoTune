# distutils: language = c
import numpy as np
cimport numpy as cnp
cimport cython
from libc.stdlib cimport malloc, free

cdef extern void dgesv_(int* n, int* nrhs,
                        double* a, int* lda,
                        int* ipiv,
                        double* b, int* ldb,
                        int* info)

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_np(cnp.ndarray[cnp.double_t, ndim=2, mode='fortran'] A not None,
             cnp.ndarray[cnp.double_t, ndim=1] b not None) -> list:
    cdef int n = A.shape[0]
    cdef int nrhs = 1
    cdef int lda = n
    cdef int ldb = n
    cdef int info
    cdef int* ipiv = <int*> malloc(n * sizeof(int))
    cdef double* a_data = <double*> A.data
    cdef double* b_data = <double*> b.data
    dgesv_(&n, &nrhs, a_data, &lda, ipiv, b_data, &ldb, &info)
    free(ipiv)
    if info != 0:
        raise ValueError(f"dgesv failed with info {info}")
    return b.tolist()
# cython build trigger