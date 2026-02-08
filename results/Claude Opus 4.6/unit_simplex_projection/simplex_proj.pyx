# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, qsort
from libc.string cimport memcpy

ctypedef np.float64_t DTYPE_t

cdef int compare_desc(const void *a, const void *b) noexcept nogil:
    cdef double va = (<double*>a)[0]
    cdef double vb = (<double*>b)[0]
    if vb > va:
        return 1
    elif vb < va:
        return -1
    return 0

def simplex_project_list(list y_list):
    cdef int n = len(y_list)
    cdef int i, rho
    cdef double cumsum_val, theta, val, theta_sum

    cdef double* buf = <double*>malloc(n * sizeof(double))
    if buf == NULL:
        raise MemoryError()

    for i in range(n):
        buf[i] = <double>y_list[i]

    cdef double* sorted_buf = <double*>malloc(n * sizeof(double))
    if sorted_buf == NULL:
        free(buf)
        raise MemoryError()
    memcpy(sorted_buf, buf, n * sizeof(double))

    qsort(sorted_buf, n, sizeof(double), compare_desc)

    cumsum_val = 0.0
    rho = 0
    for i in range(n):
        cumsum_val += sorted_buf[i]
        if sorted_buf[i] > (cumsum_val - 1.0) / (i + 1):
            rho = i

    theta_sum = 0.0
    for i in range(rho + 1):
        theta_sum += sorted_buf[i]
    theta = (theta_sum - 1.0) / (rho + 1)

    free(sorted_buf)

    result = [0.0] * n
    for i in range(n):
        val = buf[i] - theta
        result[i] = val if val > 0.0 else 0.0

    free(buf)
    return result

def simplex_project_np(np.ndarray[DTYPE_t, ndim=1] y):
    cdef int n = y.shape[0]
    cdef int i, rho
    cdef double cumsum_val, theta, val, theta_sum

    cdef np.ndarray[DTYPE_t, ndim=1] sorted_y = np.sort(y)[::-1].copy()
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.empty(n, dtype=np.float64)

    cumsum_val = 0.0
    rho = 0
    for i in range(n):
        cumsum_val += sorted_y[i]
        if sorted_y[i] > (cumsum_val - 1.0) / (i + 1):
            rho = i

    theta_sum = 0.0
    for i in range(rho + 1):
        theta_sum += sorted_y[i]
    theta = (theta_sum - 1.0) / (rho + 1)

    for i in range(n):
        val = y[i] - theta
        x[i] = val if val > 0.0 else 0.0

    return x