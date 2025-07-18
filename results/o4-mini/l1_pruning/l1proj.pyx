# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, qsort
from libc.math cimport fabs

cdef int cmp_double(const void *p1, const void *p2):
    cdef double d1 = (<double *>p1)[0]
    cdef double d2 = (<double *>p2)[0]
    if d1 < d2:
        return 1
    elif d1 > d2:
        return -1
    else:
        return 0

def project_l1(np.ndarray[np.double_t, ndim=1] v not None, double k):
    cdef Py_ssize_t n = v.shape[0]
    cdef np.ndarray[np.double_t, ndim=1] u = np.empty(n, dtype=np.double)
    cdef double *mu
    cdef Py_ssize_t i
    cdef double s = 0.0, theta = 0.0, t = 0.0, cum = 0.0

    # build absolute array and total sum
    for i in range(n):
        u[i] = fabs(v[i])
        s += u[i]
    # if inside L1 ball, copy input
    if s <= k:
        cdef np.ndarray[np.double_t, ndim=1] w = np.empty(n, dtype=np.double)
        for i in range(n):
            w[i] = v[i]
        return w

    # allocate temp for sorting
    mu = <double *>malloc(n * sizeof(double))
    if not mu:
        raise MemoryError()
    for i in range(n):
        mu[i] = u[i]
    # descending sort
    qsort(mu, n, sizeof(double), cmp_double)

    cum = 0.0
    theta = 0.0
    for i in range(n):
        cum += mu[i]
        t = (cum - k) / (i + 1)
        if mu[i] < t:
            theta = t
            break
    free(mu)

    # build output with soft-threshold
    cdef np.ndarray[np.double_t, ndim=1] w = np.empty(n, dtype=np.double)
    for i in range(n):
        ui = u[i] - theta
        if ui > 0.0:
            if v[i] >= 0.0:
                w[i] = ui
            else:
                w[i] = -ui
        else:
            w[i] = 0.0
    return w