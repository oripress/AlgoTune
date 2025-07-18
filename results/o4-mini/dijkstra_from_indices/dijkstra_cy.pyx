# cython: language_level=3
from libc.stdlib cimport malloc, free
from libcpp.priority_queue cimport priority_queue
from libcpp.utility cimport make_pair
from libcpp.pair cimport pair as cpp_pair
import cython
import numpy as np
cimport numpy as np

ctypedef np.float64_t DTYPE_t
ctypedef np.intp_t INTP_t

@cython.boundscheck(False)
@cython.wraparound(False)
def multi_source_dijkstra(np.ndarray[INTP_t, ndim=1] indptr not None,
                          np.ndarray[INTP_t, ndim=1] indices not None,
                          np.ndarray[DTYPE_t, ndim=1] data not None,
                          INTP_t n,
                          list sources):
    """
    Multi-source Dijkstra algorithm using CSR representation.
    Returns a Python list of distances, with None for unreachable nodes.
    """
    cdef DTYPE_t INF = float('inf')
    cdef DTYPE_t *dist = <DTYPE_t *> malloc(n * sizeof(DTYPE_t))
    cdef bint *vis = <bint *> malloc(n * sizeof(bint))
    cdef INTP_t i, u, v, start, end
    # Initialize distances
    for i in range(n):
        dist[i] = INF
        vis[i] = False

    # Priority queue of (distance, node)
    cdef priority_queue[cpp_pair[DTYPE_t, INTP_t]] pq
    for src in sources:
        u = <INTP_t> src
        dist[u] = 0.0
        pq.push(make_pair(<DTYPE_t>0.0, u))

    # Main loop
    cdef cpp_pair[DTYPE_t, INTP_t] top
    while pq.size() > 0:
        top = pq.top()
        pq.pop()
        u = top.second
        if top.first > dist[u]:
            continue
        start = indptr[u]
        end = indptr[u+1]
        for i in range(start, end):
            v = indices[i]
            cdef DTYPE_t nd = top.first + data[i]
            if nd < dist[v]:
                dist[v] = nd
                pq.push(make_pair(nd, v))

    # Build result list
    cdef list res = [None] * n
    for i in range(n):
        if dist[i] == INF:
            res[i] = None
        else:
            res[i] = dist[i]

    free(dist)
    free(vis)
    return res