# distutils: language = c++
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import cython
cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort
from libc.math cimport log

@cython.boundscheck(False)
@cython.wraparound(False)
def waterfill(np.ndarray[np.double_t, ndim=1] alpha not None, double P_total):
    cdef Py_ssize_t n = alpha.shape[0]
    cdef vector[double] a_vec
    cdef Py_ssize_t i
    cdef double sum_val = 0.0, w = 0.0, w_candidate = 0.0
    cdef double tot = 0.0, diff = 0.0, capacity = 0.0, scale = 0.0
    cdef np.ndarray[np.double_t, ndim=1] x

    # Copy alpha into vector and sort
    a_vec.reserve(n)
    for i in range(n):
        a_vec.push_back(alpha[i])
    sort(a_vec.begin(), a_vec.end())

    # Compute water level
    for i in range(n):
        sum_val += a_vec[i]
        w_candidate = (P_total + sum_val) / (i + 1)
        if i == n - 1 or w_candidate <= a_vec[i + 1]:
            w = w_candidate
            break

    # Allocate output and compute allocations + capacity
    x = np.empty(n, dtype=np.double)
    for i in range(n):
        diff = w - alpha[i]
        if diff > 0.0:
            x[i] = diff
            tot += diff
        else:
            x[i] = 0.0
        capacity += log(alpha[i] + x[i])

    # Scale to exact budget
    if tot > 0.0:
        scale = P_total / tot
        for i in range(n):
            x[i] *= scale

    return x, capacity