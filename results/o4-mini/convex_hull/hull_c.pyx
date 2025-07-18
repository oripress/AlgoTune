# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Free

cdef inline double cross(double x1, double y1, double x2, double y2, double x3, double y3):
    return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

cdef inline void swap_int(int *arr, int i, int j):
    cdef int tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp

cdef int partition(int *idxarr, double[:, :] A, int lo, int hi):
    cdef int mid = lo + ((hi - lo) >> 1)
    cdef int pivot_idx = idxarr[mid]
    cdef double pivot_x = A[pivot_idx, 0]
    cdef double pivot_y = A[pivot_idx, 1]
    swap_int(idxarr, mid, hi)
    cdef int store = lo
    cdef int j, cur
    for j in range(lo, hi):
        cur = idxarr[j]
        cdef double xj = A[cur, 0]
        cdef double yj = A[cur, 1]
        if xj < pivot_x or (xj == pivot_x and yj < pivot_y):
            swap_int(idxarr, store, j)
            store += 1
    swap_int(idxarr, store, hi)
    return store

cdef void quicksort_idx(int *idxarr, double[:, :] A, int lo, int hi):
    if lo < hi:
        cdef int p = partition(idxarr, A, lo, hi)
        quicksort_idx(idxarr, A, lo, p - 1)
        quicksort_idx(idxarr, A, p + 1, hi)

def convex_hull(np.ndarray[np.double_t, ndim=2] arr):
    cdef int n = arr.shape[0]
    if n <= 2:
        cdef np.ndarray[np.int64_t, ndim=1] out = np.arange(n, dtype=np.int64)
        return out
    cdef double[:, :] A = arr
    cdef int *idxarr = <int *>PyMem_Malloc(n * sizeof(int))
    cdef int i
    for i in range(n):
        idxarr[i] = i
    quicksort_idx(idxarr, A, 0, n - 1)
    cdef int *lower = <int *>PyMem_Malloc(n * sizeof(int))
    cdef int *upper = <int *>PyMem_Malloc(n * sizeof(int))
    cdef int lower_size = 0, upper_size = 0, i1, i2, cur
    # build lower hull
    for i in range(n):
        cur = idxarr[i]
        while lower_size >= 2:
            i1 = lower[lower_size - 2]
            i2 = lower[lower_size - 1]
            if cross(A[i1, 0], A[i1, 1], A[i2, 0], A[i2, 1], A[cur, 0], A[cur, 1]) <= 0:
                lower_size -= 1
            else:
                break
        lower[lower_size] = cur
        lower_size += 1
    # build upper hull
    for i in range(n - 1, -1, -1):
        cur = idxarr[i]
        while upper_size >= 2:
            i1 = upper[upper_size - 2]
            i2 = upper[upper_size - 1]
            if cross(A[i1, 0], A[i1, 1], A[i2, 0], A[i2, 1], A[cur, 0], A[cur, 1]) <= 0:
                upper_size -= 1
            else:
                break
        upper[upper_size] = cur
        upper_size += 1
    cdef int hull_n = lower_size + upper_size - 2
    cdef np.ndarray[np.int64_t, ndim=1] hull = np.empty(hull_n, dtype=np.int64)
    cdef int k = 0
    for i in range(lower_size - 1):
        hull[k] = lower[i]
        k += 1
    for i in range(upper_size - 1):
        hull[k] = upper[i]
        k += 1
    PyMem_Free(idxarr)
    PyMem_Free(lower)
    PyMem_Free(upper)
    return hull