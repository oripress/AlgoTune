# cy_wasserstein.pyx
# Cython implementation of 1D Wasserstein distance via CDF differences

import numpy as np
cimport numpy as cnp

# Disable bounds-checking and wraparound for speed
# cython: boundscheck=False, wraparound=False

def wass_c(cnp.ndarray[cnp.double_t, ndim=1] u not None,
           cnp.ndarray[cnp.double_t, ndim=1] v not None):
    cdef Py_ssize_t n = u.shape[0]
    cdef double su = 0.0, sv = 0.0
    cdef double uf, vf
    cdef double c = 0.0, total = 0.0
    cdef Py_ssize_t i
    # compute total masses
    for i in range(n):
        su += u[i]
        sv += v[i]
    uf = 1.0 / su if su != 0.0 else 0.0
    vf = 1.0 / sv if sv != 0.0 else 0.0
    # accumulate CDF difference
    for i in range(n):
        c += u[i] * uf - v[i] * vf
        total += c if c >= 0.0 else -c
    return total