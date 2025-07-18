# distutils: language=c
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef inline double cross(double* o, double* a, double* b) nogil:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull_cython(double[:,:] points_view):
    cdef int n = points_view.shape[0]
    if n <= 1:
        return np.arange(n, dtype=np.int64)
    
    # Allocate memory for hull indices
    cdef int* hull = <int*> malloc(n * sizeof(int))
    if not hull:
        raise MemoryError()
    cdef int k = 0
    
    # Build lower hull
    cdef int i
    for i in range(n):
        while k >= 2:
            if cross(&points_view[hull[k-2], 
                     &points_view[hull[k-1]], 
                     &points_view[i]) <= 0:
                k -= 1
            else:
                break
        hull[k] = i
        k += 1
    
    # Build upper hull
    cdef int t = k + 1
    for i in range(n-2, -1, -1):
        while k >= t:
            if cross(&points_view[hull[k-2]], 
                     &points_view[hull[k-1]], 
                     &points_view[i]) <= 0:
                k -= 1
            else:
                break
        hull[k] = i
        k += 1
    
    # Copy results to numpy array
    cdef np.ndarray hull_arr = np.empty(k-1, dtype=np.int64)
    for i in range(k-1):
        hull_arr[i] = hull[i]
    
    free(hull)
    return hull_arr