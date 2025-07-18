# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False, language_level=3
cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport floor, fabs, cos, sin, M_PI

cdef inline double cubic_kernel(double x):
    cdef double ax = fabs(x)
    if ax <= 1.0:
        return (4.0 - 6.0 * ax * ax + 3.0 * ax * ax * ax) / 6.0
    elif ax < 2.0:
        cdef double t = 2.0 - ax
        return (t * t * t) / 6.0
    else:
        return 0.0

def rotate_cy(np.ndarray[np.float64_t, ndim=2] arr, double angle):
    cdef int n_rows = arr.shape[0], n_cols = arr.shape[1]
    cdef double center_i = (n_rows - 1) / 2.0, center_j = (n_cols - 1) / 2.0
    cdef double theta = angle * M_PI / 180.0
    cdef double cos_t = cos(theta), sin_t = sin(theta)
    cdef int i, j, di, dj, ii, jj, floor_i, floor_j
    cdef double i_rel, j_rel, src_i, src_j, wi, wj, val
    cdef np.ndarray[np.float64_t, ndim=2] out = np.empty((n_rows, n_cols), dtype=np.float64)
    for i in range(n_rows):
        i_rel = i - center_i
        for j in range(n_cols):
            j_rel = j - center_j
            src_i = cos_t * i_rel + sin_t * j_rel + center_i
            src_j = -sin_t * i_rel + cos_t * j_rel + center_j
            floor_i = <int>floor(src_i)
            floor_j = <int>floor(src_j)
            val = 0.0
            for di in range(-1, 3):
                ii = floor_i + di
                if ii < 0 or ii >= n_rows:
                    continue
                wi = cubic_kernel(src_i - ii)
                for dj in range(-1, 3):
                    jj = floor_j + dj
                    if jj < 0 or jj >= n_cols:
                        continue
                    wj = cubic_kernel(src_j - jj)
                    val += arr[ii, jj] * wi * wj
            out[i, j] = val
    return out