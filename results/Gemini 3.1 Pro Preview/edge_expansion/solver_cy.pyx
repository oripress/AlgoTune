import cython
from libc.stdlib cimport calloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_edge_expansion(list adj_list, list nodes_S_list, Py_ssize_t n):
    cdef Py_ssize_t cut_size = 0
    cdef Py_ssize_t u, v, i, num_neighbors
    cdef list neighbors
    cdef Py_ssize_t len_S = len(nodes_S_list)
    cdef Py_ssize_t min_size
    cdef char u_in_S
    
    cdef char* in_S = <char*>calloc(n, sizeof(char))
    if not in_S:
        raise MemoryError()
        
    try:
        for u in nodes_S_list:
            in_S[u] = 1
            
        for u in range(n):
            neighbors = <list>adj_list[u]
            u_in_S = in_S[u]
            num_neighbors = len(neighbors)
            for i in range(num_neighbors):
                v = neighbors[i]
                cut_size += (u_in_S ^ in_S[v])
                    
        min_size = len_S if len_S < n - len_S else n - len_S
        return float(cut_size) / float(min_size)
    finally:
        free(in_S)