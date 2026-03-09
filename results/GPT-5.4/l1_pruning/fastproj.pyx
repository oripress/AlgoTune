# distutils: language = c++

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort as cpp_sort

cpdef cnp.ndarray project_list(list v_list, double k):
    cdef Py_ssize_t n = len(v_list)
    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef vector[double] absvals
    cdef Py_ssize_t i, pos, count
    cdef double x, val, cumsum, theta, t

    absvals.reserve(n)
    for i in range(n):
        x = float(v_list[i])
        out[i] = x
        absvals.push_back(fabs(x))

    cpp_sort(absvals.begin(), absvals.end())

    cumsum = 0.0
    theta = 0.0
    count = 0
    for pos in range(n - 1, -1, -1):
        count += 1
        val = absvals[pos]
        cumsum += val
        t = (cumsum - k) / count
        if val < t:
            theta = t
            break

    for i in range(n):
        x = out[i]
        if x > theta:
            out[i] = x - theta
        elif x < -theta:
            out[i] = x + theta
        else:
            out[i] = 0.0

    return out