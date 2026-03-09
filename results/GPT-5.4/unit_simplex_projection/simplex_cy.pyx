# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

from cpython.float cimport PyFloat_AsDouble
from cpython.object cimport PyObject
from cpython.sequence cimport PySequence_Fast, PySequence_Fast_GET_ITEM, PySequence_Fast_GET_SIZE
from libc.stdlib cimport free, malloc, qsort

ctypedef cnp.float64_t DTYPE_t

cdef int _cmp_double(const void* a, const void* b) noexcept nogil:
    cdef double da = (<double*>a)[0]
    cdef double db = (<double*>b)[0]
    return (da > db) - (da < db)

cdef cnp.ndarray[DTYPE_t, ndim=1] _project_array(object y_obj):
    cdef cnp.ndarray[DTYPE_t, ndim=1] y
    cdef cnp.ndarray[DTYPE_t, ndim=1] s
    cdef cnp.ndarray[DTYPE_t, ndim=1] out
    cdef Py_ssize_t n
    cdef Py_ssize_t i, j
    cdef double cssv, theta, t, v

    y = np.ravel(np.asarray(y_obj, dtype=np.float64))
    n = y.shape[0]

    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        out = np.empty(1, dtype=np.float64)
        out[0] = 1.0
        return out

    s = np.array(y, copy=True)
    s.sort()

    cssv = 0.0
    theta = 0.0
    for j in range(n):
        i = n - 1 - j
        cssv += s[i]
        t = (cssv - 1.0) / (j + 1)
        if s[i] > t:
            theta = t

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = y[i] - theta
        out[i] = v if v > 0.0 else 0.0
    return out

cdef cnp.ndarray[DTYPE_t, ndim=1] _project_seq(object y_obj):
    cdef object seq = PySequence_Fast(y_obj, "expected a sequence")
    cdef Py_ssize_t n = PySequence_Fast_GET_SIZE(seq)
    cdef cnp.ndarray[DTYPE_t, ndim=1] out
    cdef Py_ssize_t i, j
    cdef PyObject* item
    cdef double* s
    cdef double cssv, theta, t, v

    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        out = np.empty(1, dtype=np.float64)
        out[0] = 1.0
        return out

    out = np.empty(n, dtype=np.float64)
    s = <double*>malloc(n * sizeof(double))
    if s == NULL:
        raise MemoryError()

    try:
        for i in range(n):
            item = PySequence_Fast_GET_ITEM(seq, i)
            v = PyFloat_AsDouble(<object>item)
            s[i] = v
            out[i] = v

        with nogil:
            qsort(s, n, sizeof(double), _cmp_double)

        cssv = 0.0
        theta = 0.0
        for j in range(n):
            i = n - 1 - j
            cssv += s[i]
            t = (cssv - 1.0) / (j + 1)
            if s[i] > t:
                theta = t

        for i in range(n):
            v = out[i] - theta
            out[i] = v if v > 0.0 else 0.0

        return out
    finally:
        free(s)

def project(object y_obj):
    if isinstance(y_obj, np.ndarray):
        return _project_array(y_obj)
    return _project_seq(y_obj)