# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int find(DTYPE_t[:] parent, int x) nogil:
    """Find with path halving for better performance."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # Path halving
        x = parent[x]
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int count_components(DTYPE_t[:, :] edges, int num_nodes):
    """Count connected components using optimized Union-Find."""
    cdef int i, u, v, root_u, root_v
    cdef int num_edges = edges.shape[0]
    cdef int components = num_nodes  # Start with all nodes as separate components
    
    # Initialize parent array
    cdef np.ndarray[DTYPE_t, ndim=1] parent = np.arange(num_nodes, dtype=np.int32)
    cdef DTYPE_t[:] parent_view = parent
    
    # Process edges and track component count
    with nogil:
        for i in range(num_edges):
            u = edges[i, 0]
            v = edges[i, 1]
            
            root_u = find(parent_view, u)
            root_v = find(parent_view, v)
            
            if root_u != root_v:
                # Union - always attach smaller index to larger for consistency
                if root_u < root_v:
                    parent_view[root_v] = root_u
                else:
                    parent_view[root_u] = root_v
                components -= 1  # Two components merged into one
    
    return components