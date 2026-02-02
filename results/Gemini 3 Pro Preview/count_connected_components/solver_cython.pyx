# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True

from libc.stdlib cimport malloc, free

def solve_cython(int n, list edges):
    cdef int* parent
    cdef int i, u, v, root_u, root_v
    cdef int num_components = n
    
    if n == 0:
        return 0
        
    parent = <int*> malloc(n * sizeof(int))
    if not parent:
        raise MemoryError()
        
    for i in range(n):
        parent[i] = i
        
    for u, v in edges:
        root_u = u
        while root_u != parent[root_u]:
            # Path compression (halving)
            parent[root_u] = parent[parent[root_u]]
            root_u = parent[root_u]
            
        root_v = v
        while root_v != parent[root_v]:
            # Path compression (halving)
            parent[root_v] = parent[parent[root_v]]
            root_v = parent[root_v]
            
        if root_u != root_v:
            parent[root_u] = root_v
            num_components -= 1
            if num_components == 1:
                break
                
    free(parent)
    return num_components