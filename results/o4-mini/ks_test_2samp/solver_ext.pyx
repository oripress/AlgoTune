# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
from libc.math cimport exp

def ks_stat(double[:] x, double[:] y):
    """
    Compute two-sample KS statistic (two-sided) for sorted arrays x, y.
    """
    cdef Py_ssize_t n1 = x.shape[0]
    cdef Py_ssize_t n2 = y.shape[0]
    cdef double inv_n1 = 1.0 / n1
    cdef double inv_n2 = 1.0 / n2
    cdef Py_ssize_t i = 0, j = 0
    cdef double d = 0.0
    cdef double diff
    # merge-like scan
    while i < n1 and j < n2:
        if x[i] <= y[j]:
            i += 1
        else:
            j += 1
        diff = i * inv_n1 - j * inv_n2
        if diff < 0.0:
            diff = -diff
        if diff > d:
            d = diff
    # finish tails
    while i < n1:
        i += 1
        diff = i * inv_n1 - j * inv_n2
        if diff < 0.0:
            diff = -diff
        if diff > d:
            d = diff
    while j < n2:
        j += 1
        diff = i * inv_n1 - j * inv_n2
        if diff < 0.0:
            diff = -diff
        if diff > d:
            d = diff
    return d

def ks_sf(double t):
    """
    Asymptotic survival function: P(sqrt(n_eff)*D > t)
    using series 2*sum_{k>=1}(-1)^{k-1} exp(-2*k^2*t^2).
    """
    cdef double total = 0.0
    cdef double two_t2 = 2.0 * t * t
    cdef double sign = 1.0
    cdef double term
    cdef int k = 1
    cdef double tol = 1e-15
    while True:
        term = sign * exp(-two_t2 * k * k)
        total += term
        if term < 0.0:
            if -term < tol:
                break
        else:
            if term < tol:
                break
        sign = -sign
        k += 1
    return 2.0 * total