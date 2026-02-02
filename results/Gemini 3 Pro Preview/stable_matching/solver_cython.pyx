import numpy as np
cimport numpy as np
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_cython(list proposer_prefs_list, list receiver_prefs_list, int n):
    # Allocate arrays
    cdef int[:, ::1] prop_prefs = np.empty((n, n), dtype=np.int32)
    cdef int[:, ::1] recv_rank = np.empty((n, n), dtype=np.int32)
    
    cdef int i, j, p, rank
    cdef list row
    
    # Fill prop_prefs
    for i in range(n):
        row = proposer_prefs_list[i]
        for j in range(n):
            prop_prefs[i, j] = row[j]

    # Fill recv_rank
    # receiver_prefs_list[i] is the preference list for receiver i
    # if receiver_prefs_list[i][rank] == p, then recv_rank[i][p] = rank
    for i in range(n):
        row = receiver_prefs_list[i]
        for rank in range(n):
            p = row[rank]
            recv_rank[i, p] = rank
            
    cdef int[::1] next_prop_idx = np.zeros(n, dtype=np.int32)
    cdef int[::1] recv_match = np.full(n, -1, dtype=np.int32)
    
    # Stack for free proposers
    cdef int[::1] free_props = np.arange(n, dtype=np.int32)
    cdef int free_count = n
    
    cdef int r, cur
    
    while free_count > 0:
        free_count -= 1
        p = free_props[free_count]
        
        # Get next receiver
        r = prop_prefs[p, next_prop_idx[p]]
        next_prop_idx[p] += 1
        
        cur = recv_match[r]
        
        if cur == -1:
            recv_match[r] = p
        else:
            if recv_rank[r, p] < recv_rank[r, cur]:
                recv_match[r] = p
                free_props[free_count] = cur
                free_count += 1
            else:
                free_props[free_count] = p
                free_count += 1
                
    # Construct result
    cdef int[::1] matching = np.empty(n, dtype=np.int32)
    for r in range(n):
        p = recv_match[r]
        matching[p] = r
        
    return np.asarray(matching).tolist()