# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue
from cython.parallel import prange

ctypedef pair[double, int] pdi

def dijkstra_cython(
    double[:] data,
    int[:] indices,
    int[:] indptr,
    int num_nodes,
    long[:] source_indices,
    int num_threads
):
    cdef int num_sources = source_indices.shape[0]
    cdef int i
    
    # Result array: (num_sources, num_nodes)
    cdef double[:, :] distances = np.full((num_sources, num_nodes), np.inf, dtype=np.float64)
    
    for i in prange(num_sources, nogil=True, num_threads=num_threads):
        run_dijkstra_single(data, indices, indptr, num_nodes, source_indices[i], distances[i])
                    
    return np.asarray(distances)

cdef void run_dijkstra_single(
    double[:] data,
    int[:] indices,
    int[:] indptr,
    int num_nodes,
    int src,
    double[:] dist_row
) noexcept nogil:
    cdef int u, v, idx
    cdef double weight, dist_u, new_dist
    cdef priority_queue[pdi] pq
    
    dist_row[src] = 0.0
    # Use negative distance for max-heap to act as min-heap
    pq.push(pdi(-0.0, src))
    
    while not pq.empty():
        dist_u = -pq.top().first
        u = pq.top().second
        pq.pop()
        
        if dist_u > dist_row[u]:
            continue
        
        for idx in range(indptr[u], indptr[u+1]):
            v = indices[idx]
            weight = data[idx]
            new_dist = dist_u + weight
            
            if new_dist < dist_row[v]:
                dist_row[v] = new_dist
                pq.push(pdi(-new_dist, v))