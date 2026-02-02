# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from cpython.ref cimport PyObject
from libcpp.pair cimport pair
from cython.parallel import prange

cdef extern from "<queue>" namespace "std":
    cdef cppclass priority_queue[T]:
        priority_queue() nogil
        void push(T&) nogil
        T& top() nogil
        void pop() nogil
        bint empty() nogil

ctypedef pair[double, int] pdi

def parallel_dijkstra(
    int n_nodes,
    double[:] data,
    int[:] indices,
    int[:] indptr,
    double[:] data_T,
    int[:] indices_T,
    int[:] indptr_T,
    int n_threads
):
    cdef double[:, ::1] dist_matrix = np.empty((n_nodes, n_nodes), dtype=np.float64)
    cdef int i, j
    
    # Initialize with infinity in parallel
    for i in prange(n_nodes, nogil=True, num_threads=n_threads):
        for j in range(n_nodes):
            dist_matrix[i, j] = 1.0/0.0 # Infinity

    for i in prange(n_nodes, nogil=True, num_threads=n_threads, schedule='dynamic'):
        dijkstra_single_source(
            i, n_nodes,
            data, indices, indptr,
            data_T, indices_T, indptr_T,
            dist_matrix[i]
        )
        
    return convert_to_list(dist_matrix, n_nodes)

cdef list convert_to_list(double[:, ::1] dist_matrix, int n_nodes):
    cdef list result = []
    cdef list row_list
    cdef int i, j
    cdef double val
    
    for i in range(n_nodes):
        row_list = []
        for j in range(n_nodes):
            val = dist_matrix[i, j]
            if val == 1.0/0.0: # Check for infinity
                row_list.append(None)
            else:
                row_list.append(val)
        result.append(row_list)
    return result

cdef void dijkstra_single_source(
    int start_node,
    int n_nodes,
    double[:] data,
    int[:] indices,
    int[:] indptr,
    double[:] data_T,
    int[:] indices_T,
    int[:] indptr_T,
    double[:] dist_row
) noexcept nogil:
    cdef priority_queue[pdi] pq
    cdef int u, v, idx
    cdef double d, weight, new_dist
    cdef pdi item
    
    dist_row[start_node] = 0.0
    item = pdi(0.0, start_node)
    pq.push(item)
    
    while not pq.empty():
        item = pq.top()
        d = -item.first
        u = item.second
        pq.pop()
        
        if d > dist_row[u]:
            continue
        
        # Neighbors in G
        for idx in range(indptr[u], indptr[u+1]):
            v = indices[idx]
            weight = data[idx]
            new_dist = d + weight
            if new_dist < dist_row[v]:
                dist_row[v] = new_dist
                item = pdi(-new_dist, v)
                pq.push(item)
                
        # Neighbors in G_T
        for idx in range(indptr_T[u], indptr_T[u+1]):
            v = indices_T[idx]
            weight = data_T[idx]
            new_dist = d + weight
            if new_dist < dist_row[v]:
                dist_row[v] = new_dist
                item = pdi(-new_dist, v)
                pq.push(item)