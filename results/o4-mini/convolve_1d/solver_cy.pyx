# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as cnp

cpdef cnp.ndarray[double, ndim=1] direct_conv(double[::1] a, double[::1] b):
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t m = b.shape[0]
    cdef Py_ssize_t total = n + m - 1
    cdef Py_ssize_t i, j
    cdef cnp.ndarray[double, ndim=1] result = np.zeros(total, dtype=np.float64)
    cdef double* a_ptr = &a[0]
    cdef double* b_ptr = &b[0]
    cdef double* r_ptr = &result[0]
    for i in range(n):
        for j in range(m):
            r_ptr[i + j] += a_ptr[i] * b_ptr[j]
    return result