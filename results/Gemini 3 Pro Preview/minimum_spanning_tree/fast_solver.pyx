import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int32_t INT32_t
ctypedef np.float64_t FLOAT64_t

@cython.boundscheck(False)
@cython.wraparound(False)
def solve_cython(int num_nodes, 
                 INT32_t[:] u_in, 
                 INT32_t[:] v_in, 
                 FLOAT64_t[:] w_in):
    
    cdef int num_edges = u_in.shape[0]
    cdef int i, u, v, idx
    
    # Adjacency list construction
    # We need to preserve insertion order.
    # head[u] points to the first edge added to u.
    # tail[u] points to the last edge added to u.
    # next_entry[ptr] points to the next edge in the list.
    
    cdef INT32_t[:] head = np.full(num_nodes, -1, dtype=np.int32)
    cdef INT32_t[:] tail = np.full(num_nodes, -1, dtype=np.int32)
    
    # Each input edge creates 2 entries in the adjacency list.
    cdef int capacity = 2 * num_edges
    cdef INT32_t[:] next_entry = np.full(capacity, -1, dtype=np.int32)
    cdef INT32_t[:] to_node = np.empty(capacity, dtype=np.int32)
    cdef INT32_t[:] original_idx = np.empty(capacity, dtype=np.int32)
    
    cdef int ptr = 0
    
    for i in range(num_edges):
        u = u_in[i]
        v = v_in[i]
        
        # Add u -> v
        to_node[ptr] = v
        original_idx[ptr] = i
        
        if head[u] == -1:
            head[u] = ptr
        else:
            next_entry[tail[u]] = ptr
        tail[u] = ptr
        ptr += 1
        
        # Add v -> u
        to_node[ptr] = u
        original_idx[ptr] = i
        
        if head[v] == -1:
            head[v] = ptr
        else:
            next_entry[tail[v]] = ptr
        tail[v] = ptr
        ptr += 1
        
    # Collect edges in NetworkX order
    # Iterate u 0..N-1
    # Iterate neighbors. If u < neighbor, add original_idx.
    
    cdef INT32_t[:] candidates = np.empty(num_edges, dtype=np.int32)
    cdef int count = 0
    cdef int curr
    cdef int neighbor
    
    for u in range(num_nodes):
        curr = head[u]
        while curr != -1:
            neighbor = to_node[curr]
            if u < neighbor:
                candidates[count] = original_idx[curr]
                count += 1
            curr = next_entry[curr]
            
    # Stable sort based on weights
    cdef np.ndarray[INT32_t, ndim=1] candidates_np = np.asarray(candidates)[:count]
    cdef np.ndarray[FLOAT64_t, ndim=1] weights_subset = np.asarray(w_in)[candidates_np]
    
    # Use numpy's stable argsort
    cdef np.ndarray[np.int64_t, ndim=1] sort_order = np.argsort(weights_subset, kind='stable')
    cdef INT32_t[:] sorted_indices = candidates_np[sort_order]
    
    # Kruskal's Algorithm
    cdef INT32_t[:] parent = np.arange(num_nodes, dtype=np.int32)
    cdef INT32_t[:] rank = np.zeros(num_nodes, dtype=np.int32)
    
    cdef int root_u, root_v, tmp
    cdef int edges_found = 0
    cdef int u_orig, v_orig
    cdef double w_orig
    
    mst_edges = []
    
    for i in range(count):
        idx = sorted_indices[i]
        u = u_in[idx]
        v = v_in[idx]
        
        # Find u
        root_u = u
        while root_u != parent[root_u]:
            root_u = parent[root_u]
        # Path compression u
        while u != root_u:
            tmp = parent[u]
            parent[u] = root_u
            u = tmp
            
        # Find v
        root_v = v
        while root_v != parent[root_v]:
            root_v = parent[root_v]
        # Path compression v
        while v != root_v:
            tmp = parent[v]
            parent[v] = root_v
            v = tmp
            
        if root_u != root_v:
            if rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            elif rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            else:
                parent[root_u] = root_v
                rank[root_v] += 1
            
            # Add to result
            u_orig = u_in[idx]
            v_orig = v_in[idx]
            w_orig = w_in[idx]
            
            if u_orig > v_orig:
                mst_edges.append([v_orig, u_orig, w_orig])
            else:
                mst_edges.append([u_orig, v_orig, w_orig])
                
            edges_found += 1
            if edges_found == num_nodes - 1:
                break
                
    return mst_edges