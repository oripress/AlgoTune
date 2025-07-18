#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

ctypedef np.int8_t DTYPE_t
ctypedef np.uint8_t BOOL_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list fast_greedy_dominating_set(np.ndarray[DTYPE_t, ndim=2] adj, int n):
    """Fast greedy algorithm for dominating set using Cython."""
    cdef np.ndarray[BOOL_t, ndim=1] dominated = np.zeros(n, dtype=np.uint8)
    cdef list dominating_set = []
    cdef int v, u, count, best_vertex, best_count
    cdef int total_dominated = 0
    
    while total_dominated < n:
        best_vertex = -1
        best_count = 0
        
        for v in range(n):
            if v in dominating_set:
                continue
                
            count = 0
            if dominated[v] == 0:
                count += 1
            
            for u in range(n):
                if adj[v, u] == 1 and dominated[u] == 0:
                    count += 1
            
            if count > best_count:
                best_count = count
                best_vertex = v
        
        if best_vertex == -1:
            break
            
        dominating_set.append(best_vertex)
        
        if dominated[best_vertex] == 0:
            dominated[best_vertex] = 1
            total_dominated += 1
            
        for u in range(n):
            if adj[best_vertex, u] == 1 and dominated[u] == 0:
                dominated[u] = 1
                total_dominated += 1
    
    return dominating_set

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef list exact_dominating_set_small(np.ndarray[DTYPE_t, ndim=2] adj, int n):
    """Exact algorithm for small graphs using bit manipulation."""
    cdef int min_size = n + 1
    cdef list best_set = []
    cdef int subset, size, i, j
    cdef bint valid
    cdef int dominated_mask
    
    # Try all possible subsets
    for subset in range(1, 1 << n):
        size = 0
        for i in range(n):
            if subset & (1 << i):
                size += 1
        
        if size >= min_size:
            continue
        
        # Check if this subset is a dominating set
        dominated_mask = 0
        
        for i in range(n):
            if subset & (1 << i):
                # i is in the dominating set
                dominated_mask |= (1 << i)
                # Add all neighbors of i
                for j in range(n):
                    if adj[i, j] == 1:
                        dominated_mask |= (1 << j)
        
        # Check if all vertices are dominated
        if dominated_mask == (1 << n) - 1:
            if size < min_size:
                min_size = size
                best_set = []
                for i in range(n):
                    if subset & (1 << i):
                        best_set.append(i)
    
    return best_set

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple preprocess_graph(np.ndarray[DTYPE_t, ndim=2] adj, int n):
    """Preprocess graph to identify forced vertices and reduce problem size."""
    cdef np.ndarray[np.int32_t, ndim=1] degrees = np.zeros(n, dtype=np.int32)
    cdef list forced = []
    cdef list isolated = []
    cdef int i, j
    
    # Calculate degrees
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                degrees[i] += 1
    
    # Find isolated vertices (must be in dominating set)
    for i in range(n):
        if degrees[i] == 0:
            isolated.append(i)
    
    # Find vertices that must be in dominating set
    # (vertices whose neighbors don't connect to each other)
    for i in range(n):
        if degrees[i] > 0:
            must_include = True
            neighbors = []
            for j in range(n):
                if adj[i, j] == 1:
                    neighbors.append(j)
            
            # Check if neighbors form a clique
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if adj[neighbors[j], neighbors[k]] == 0:
                        must_include = False
                        break
                if not must_include:
                    break
            
            if must_include and len(neighbors) > 1:
                # If neighbors don't form a clique, we might need this vertex
                pass
    
    return isolated, degrees

@cython.boundscheck(False) 
@cython.wraparound(False)
cpdef list improve_solution_cy(np.ndarray[DTYPE_t, ndim=2] adj, list initial_set, int n):
    """Improve solution using fast local search."""
    cdef list current_set = list(initial_set)
    cdef np.ndarray[BOOL_t, ndim=1] dominated
    cdef int i, j, v
    cdef bint improved = True
    cdef bint all_dominated
    
    while improved:
        improved = False
        
        # Try to remove each vertex
        for i in range(len(current_set)):
            v = current_set[i]
            
            # Check if removing v still gives a dominating set
            dominated = np.zeros(n, dtype=np.uint8)
            
            # Mark vertices dominated by remaining vertices
            for j in range(len(current_set)):
                if j != i:
                    u = current_set[j]
                    dominated[u] = 1
                    for k in range(n):
                        if adj[u, k] == 1:
                            dominated[k] = 1
            
            # Check if all vertices are dominated
            all_dominated = True
            for k in range(n):
                if dominated[k] == 0:
                    all_dominated = False
                    break
            
            if all_dominated:
                current_set = current_set[:i] + current_set[i+1:]
                improved = True
                break
    
    return current_set