# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.stdlib cimport malloc, free
from cpython.list cimport PyList_GET_ITEM, PyList_GET_SIZE

def count_components(int n, list edges):
    cdef int components = n
    cdef int u, v
    cdef int i
    cdef Py_ssize_t num_edges = PyList_GET_SIZE(edges)
    cdef Py_ssize_t edge_idx
    cdef tuple edge_obj
    
    cdef int* parent = <int*>malloc(n * sizeof(int))
    if not parent:
        raise MemoryError()
    
    for i in range(n):
        parent[i] = i
        
    for edge_idx in range(num_edges):
        edge_obj = <tuple>PyList_GET_ITEM(edges, edge_idx)
        u = <int>edge_obj[0]
        v = <int>edge_obj[1]
        
        if parent[u] == parent[v]:
            continue
            
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
            
        if u != v:
            if u < v:
                parent[v] = u
            else:
                parent[u] = v
            components -= 1
            if components == 1:
                break
                
    free(parent)
    return components