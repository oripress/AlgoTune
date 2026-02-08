# cython: boundscheck=False, wraparound=False, cdivision=True
from libc.stdlib cimport malloc, free

def count_components_cy(int n, list edges):
    cdef int *parent = <int *>malloc(n * sizeof(int))
    cdef int *rnk = <int *>malloc(n * sizeof(int))
    cdef int num_components = n
    cdef int u, v, ru, rv, x, i
    cdef int num_edges = len(edges)
    
    for i in range(n):
        parent[i] = i
        rnk[i] = 0
    
    for i in range(num_edges):
        edge = edges[i]
        u = <int>edge[0]
        v = <int>edge[1]
        
        # find u
        x = u
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        ru = x
        
        # find v
        x = v
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        rv = x
        
        if ru != rv:
            num_components -= 1
            if rnk[ru] < rnk[rv]:
                parent[ru] = rv
            elif rnk[ru] > rnk[rv]:
                parent[rv] = ru
            else:
                parent[rv] = ru
                rnk[ru] += 1
    
    free(parent)
    free(rnk)
    return num_components