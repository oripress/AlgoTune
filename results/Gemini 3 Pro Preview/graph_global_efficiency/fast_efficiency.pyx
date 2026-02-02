# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset

ctypedef np.int32_t int32_t

def calculate_efficiency_cython(int32_t[:] indptr, int32_t[:] indices, int n, int num_threads):
    cdef double total_inv_dist = 0.0
    cdef int i
    
    # Release GIL for parallelism
    with nogil:
        for i in prange(n, num_threads=num_threads, schedule='dynamic'):
            total_inv_dist += bfs_efficiency(i, n, &indptr[0], &indices[0])
            
    return total_inv_dist

cdef double bfs_efficiency(int start_node, int n, int32_t* indptr, int32_t* indices) nogil:
    cdef int* queue = <int*> malloc(n * sizeof(int))
    cdef int* dist = <int*> malloc(n * sizeof(int))
    cdef int head = 0
    cdef int tail = 0
    cdef int u, v, i
    cdef int d
    cdef double local_sum = 0.0
    
    if queue == NULL or dist == NULL:
        if queue != NULL: free(queue)
        if dist != NULL: free(dist)
        return 0.0 

    # Initialize dist array to -1
    memset(dist, -1, n * sizeof(int))
        
    dist[start_node] = 0
    queue[tail] = start_node
    tail += 1
    
    while head < tail:
        u = queue[head]
        head += 1
        
        d = dist[u]
        if d > 0:
            local_sum += 1.0 / d
            
        # Iterate neighbors
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if dist[v] == -1:
                dist[v] = d + 1
                queue[tail] = v
                tail += 1
                
    free(queue)
    free(dist)
    return local_sum