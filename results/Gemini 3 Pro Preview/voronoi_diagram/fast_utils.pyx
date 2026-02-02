# cython: language_level=3
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def reorder_regions(list regions, long[:] point_region):
    cdef int n = point_region.shape[0]
    cdef list result = [None] * n
    cdef int i
    cdef long idx
    
    for i in range(n):
        idx = point_region[i]
        result[i] = regions[idx]
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def convert_ridge_vertices(list ridge_vertices):
    cdef int n = len(ridge_vertices)
    cdef np.ndarray[long, ndim=2] result = np.empty((n, 2), dtype=np.int64)
    cdef int i
    cdef list ridge
    
    for i in range(n):
        ridge = ridge_vertices[i]
        result[i, 0] = ridge[0]
        result[i, 1] = ridge[1]
    return result