# cython: boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as cnp

def solve(dict problem):
    cdef int k = problem.get('k', 0)
    cdef cnp.ndarray[cnp.double_t, ndim=1] v = np.asarray(problem.get('v', []), dtype=np.float64)
    cdef Py_ssize_t n = v.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=1] abs_v
    cdef cnp.ndarray[cnp.int_t, ndim=1] idx
    cdef cnp.ndarray[cnp.double_t, ndim=1] w
    cdef Py_ssize_t left, right, target, pivot_idx, store_idx, i
    cdef double pivot
    cdef int tmp

    # trivial cases
    if k <= 0:
        w = np.zeros(n, dtype=np.float64)
        return {'solution': w}
    if k >= n:
        return {'solution': v}

    abs_v = np.abs(v)
    idx = np.arange(n, dtype=np.int32)
    left, right = 0, n - 1
    target = n - k

    # Quickselect to get top-k indices in idx[target:]
    while left < right:
        pivot_idx = (left + right) // 2
        pivot = abs_v[idx[pivot_idx]]
        # move pivot to end
        tmp = idx[pivot_idx]; idx[pivot_idx] = idx[right]; idx[right] = tmp
        store_idx = left
        for i in range(left, right):
            if abs_v[idx[i]] < pivot:
                tmp = idx[i]; idx[i] = idx[store_idx]; idx[store_idx] = tmp
                store_idx += 1
        # move pivot to its final place
        tmp = idx[store_idx]; idx[store_idx] = idx[right]; idx[right] = tmp
        if store_idx == target:
            break
        elif store_idx < target:
            left = store_idx + 1
        else:
            right = store_idx - 1

    # build solution array with top k values
    w = np.zeros(n, dtype=np.float64)
    for i in range(target, n):
        w[idx[i]] = v[idx[i]]
    return {'solution': w}