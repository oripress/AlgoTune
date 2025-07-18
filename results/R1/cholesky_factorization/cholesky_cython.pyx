import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cholesky_cython(double[:, ::1] A):
    cdef int n = A.shape[0]
    cdef double[:, ::1] L = np.zeros((n, n))
    cdef int i, j, k
    cdef double s
    
    for i in range(n):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = (A[i, i] - s) ** 0.5
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return np.asarray(L)
cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def cholesky_cython(double[:, :] A):
    cdef int n = A.shape[0]
    cdef double[:, :] L = np.zeros((n, n), dtype=np.float64)
    cdef int i, j, k
    cdef double s

    for i in range(n):
        for j in range(i+1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return np.asarray(L)