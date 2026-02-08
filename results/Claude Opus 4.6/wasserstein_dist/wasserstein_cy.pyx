# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
cimport cython

def wasserstein_from_list(list u_list, list v_list):
    cdef Py_ssize_t n = len(u_list)
    cdef double u_sum = 0.0
    cdef double v_sum = 0.0
    cdef double cdf_diff = 0.0
    cdef double result = 0.0
    cdef double inv_u, inv_v, ui, vi
    cdef Py_ssize_t i

    # Single pass: compute sums and store values
    cdef double* u_arr = <double*>malloc(n * sizeof(double))
    cdef double* v_arr = <double*>malloc(n * sizeof(double))

    if u_arr == NULL or v_arr == NULL:
        if u_arr != NULL:
            free(u_arr)
        if v_arr != NULL:
            free(v_arr)
        return 0.0

    for i in range(n):
        ui = <double>(u_list[i])
        vi = <double>(v_list[i])
        u_arr[i] = ui
        v_arr[i] = vi
        u_sum += ui
        v_sum += vi

    inv_u = 1.0 / u_sum
    inv_v = 1.0 / v_sum

    for i in range(n):
        cdf_diff += u_arr[i] * inv_u - v_arr[i] * inv_v
        if cdf_diff >= 0.0:
            result += cdf_diff
        else:
            result -= cdf_diff

    free(u_arr)
    free(v_arr)

    return result

def wasserstein_from_arrays(double[::1] u, double[::1] v):
    cdef Py_ssize_t n = u.shape[0]
    cdef double u_sum = 0.0
    cdef double v_sum = 0.0
    cdef double cdf_diff = 0.0
    cdef double result = 0.0
    cdef double inv_u, inv_v
    cdef Py_ssize_t i

    for i in range(n):
        u_sum += u[i]
        v_sum += v[i]

    inv_u = 1.0 / u_sum
    inv_v = 1.0 / v_sum

    for i in range(n):
        cdf_diff += u[i] * inv_u - v[i] * inv_v
        if cdf_diff >= 0.0:
            result += cdf_diff
        else:
            result -= cdf_diff

    return result