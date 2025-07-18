import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

def chol(np.ndarray[np.float64_t, ndim=2] A not None):
    cdef Py_ssize_t n = A.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] L = np.zeros((n, n), dtype=np.float64)
    cdef Py_ssize_t i, j, k
    cdef double s
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]
    return L