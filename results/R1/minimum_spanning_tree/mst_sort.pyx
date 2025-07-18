import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_sort_edges(np.ndarray[np.int32_t, ndim=1] rows,
                      np.ndarray[np.int32_t, ndim=1] cols,
                      np.ndarray[np.float64_t, ndim=1] data):
    cdef int n = rows.shape[0]
    cdef np.ndarray[np.int32_t, ndim=1] min_nodes = np.empty(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] max_nodes = np.empty(n, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=1] weights = np.empty(n, dtype=np.float64)
    
    cdef int i, u, v
    cdef double w
    
    # Normalize node order
    for i in range(n):
        u = rows[i]
        v = cols[i]
        w = data[i]
        if u < v:
            min_nodes[i] = u
            max_nodes[i] = v
        else:
            min_nodes[i] = v
            max_nodes[i] = u
        weights[i] = w
    
    # Create index array for sorting
    cdef np.ndarray[np.intp_t, ndim=1] idx = np.argsort(min_nodes, kind='mergesort')
    
    # Prepare final edge list
    cdef list result = []
    for i in range(n):
        j = idx[i]
        result.append([min_nodes[j], max_nodes[j], weights[j]])
    
    return result