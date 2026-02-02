# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

from libc.stdlib cimport calloc, malloc, free

cdef int binary_search(Py_ssize_t *arr, Py_ssize_t size, Py_ssize_t target) nogil:
    cdef Py_ssize_t left = 0
    cdef Py_ssize_t right = size - 1
    cdef Py_ssize_t mid
    cdef Py_ssize_t val
    
    while left <= right:
        mid = left + ((right - left) >> 1)
        val = arr[mid]
        if val == target:
            return 1
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1
    return 0

def solve_cython(list adj_list, list nodes_S):
    cdef Py_ssize_t n = len(adj_list)
    cdef Py_ssize_t len_S = len(nodes_S)
    
    if len_S == 0 or len_S == n:
        return 0.0
        
    # Convert nodes_S to C array for faster access
    cdef Py_ssize_t *S_arr = <Py_ssize_t *>malloc(len_S * sizeof(Py_ssize_t))
    if not S_arr:
        raise MemoryError()
        
    cdef Py_ssize_t i, j, n_neighbors
    cdef Py_ssize_t u, v
    cdef Py_ssize_t cut_edges = 0
    cdef list neighbors
    cdef char *mask = NULL
    cdef Py_ssize_t prev_v
    
    # Fill S_arr
    for i in range(len_S):
        S_arr[i] = nodes_S[i]
        
    # Heuristic to choose method
    cdef int use_mask = 1
    if n > 200000 and len_S < n / 50:
        use_mask = 0
        
    if use_mask:
        mask = <char *>calloc(n, sizeof(char))
        if not mask:
            free(S_arr)
            raise MemoryError()
        
        for i in range(len_S):
            u = S_arr[i]
            if u >= 0 and u < n:
                mask[u] = 1
        
        for i in range(len_S):
            u = S_arr[i]
            if u < 0 or u >= n: continue
            
            neighbors = adj_list[u]
            n_neighbors = len(neighbors)
            
            prev_v = -1
            
            for j in range(n_neighbors):
                v = neighbors[j]
                
                if v == prev_v:
                    continue
                prev_v = v
                
                if v >= 0 and v < n:
                    if mask[v] == 0:
                        cut_edges += 1
                else:
                    cut_edges += 1
        free(mask)
        
    else:
        # Use binary search
        for i in range(len_S):
            u = S_arr[i]
            if u < 0 or u >= n: continue
            
            neighbors = adj_list[u]
            n_neighbors = len(neighbors)
            
            prev_v = -1
            
            for j in range(n_neighbors):
                v = neighbors[j]
                
                if v == prev_v:
                    continue
                prev_v = v
                
                if binary_search(S_arr, len_S, v) == 0:
                    cut_edges += 1
                    
    free(S_arr)
    return float(cut_edges) / len_S