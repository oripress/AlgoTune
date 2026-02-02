# distutils: language = c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libcpp.vector cimport vector
from libcpp.algorithm cimport sort, unique
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
import numpy as np
cimport numpy as np

def find_articulation_points(int num_nodes, object edges):
    # CSR representation
    cdef vector[int] adj_indices
    cdef vector[int] adj_indptr
    cdef vector[int] degrees
    
    degrees.assign(num_nodes, 0)
    
    cdef int u, v
    cdef long[:, :] edges_view
    cdef int i_edge
    cdef int num_edges = 0
    
    # First pass: count degrees
    if isinstance(edges, np.ndarray):
        if edges.dtype != np.int64:
             edges = edges.astype(np.int64)
        edges_view = edges
        num_edges = edges_view.shape[0]
        for i_edge in range(num_edges):
            u = edges_view[i_edge, 0]
            v = edges_view[i_edge, 1]
            degrees[u] += 1
            degrees[v] += 1
    else:
        if not isinstance(edges, list):
             edges = list(edges)
        num_edges = len(edges)
        for edge in edges:
            u = edge[0]
            v = edge[1]
            degrees[u] += 1
            degrees[v] += 1

    # Build CSR structure
    adj_indptr.resize(num_nodes + 1)
    adj_indptr[0] = 0
    cdef int i_node
    for i_node in range(num_nodes):
        adj_indptr[i_node+1] = adj_indptr[i_node] + degrees[i_node]
        
    adj_indices.resize(adj_indptr[num_nodes])
    
    # Reset degrees to use as current offset
    cdef vector[int] current_offset
    current_offset.assign(num_nodes, 0)
    
    # Second pass: fill adjacency
    if isinstance(edges, np.ndarray):
        for i_edge in range(num_edges):
            u = edges_view[i_edge, 0]
            v = edges_view[i_edge, 1]
            adj_indices[adj_indptr[u] + current_offset[u]] = v
            current_offset[u] += 1
            adj_indices[adj_indptr[v] + current_offset[v]] = u
            current_offset[v] += 1
    else:
        for edge in edges:
            u = edge[0]
            v = edge[1]
            adj_indices[adj_indptr[u] + current_offset[u]] = v
            current_offset[u] += 1
            adj_indices[adj_indptr[v] + current_offset[v]] = u
            current_offset[v] += 1

    cdef vector[int] ap_list
    ap_list.reserve(num_nodes)
    
    cdef int* p_ids = <int*> malloc(num_nodes * sizeof(int))
    cdef int* p_low = <int*> malloc(num_nodes * sizeof(int))
    cdef int* p_parent = <int*> malloc(num_nodes * sizeof(int))
    cdef int* p_current_edge_ptr = <int*> malloc(num_nodes * sizeof(int))
    cdef int* p_stack = <int*> malloc(num_nodes * sizeof(int))
    cdef char* p_is_ap = <char*> malloc(num_nodes * sizeof(char))
    
    if not p_ids or not p_low or not p_parent or not p_current_edge_ptr or not p_stack or not p_is_ap:
        if p_ids: free(p_ids)
        if p_low: free(p_low)
        if p_parent: free(p_parent)
        if p_current_edge_ptr: free(p_current_edge_ptr)
        if p_stack: free(p_stack)
        if p_is_ap: free(p_is_ap)
        raise MemoryError()

    memset(p_ids, -1, num_nodes * sizeof(int))
    memset(p_parent, -1, num_nodes * sizeof(int))
    memset(p_is_ap, 0, num_nodes * sizeof(char))
    
    # Initialize current_edge_ptr with the start indices from adj_indptr
    cdef int* p_adj_indptr = adj_indptr.data()
    cdef int* p_adj_indices = adj_indices.data()
    
    memcpy(p_current_edge_ptr, p_adj_indptr, num_nodes * sizeof(int))
    
    cdef int stack_top = 0
    cdef int timer = 0
    cdef int i, root, root_children, p
    cdef int end_idx
    
    for i in range(num_nodes):
        if p_ids[i] != -1:
            continue
            
        root = i
        p_ids[root] = p_low[root] = timer
        timer += 1
        
        p_stack[stack_top] = root
        stack_top += 1
        root_children = 0
        
        while stack_top > 0:
            u = p_stack[stack_top - 1]
            
            end_idx = p_adj_indptr[u+1]
            
            if p_current_edge_ptr[u] < end_idx:
                v = p_adj_indices[p_current_edge_ptr[u]]
                p_current_edge_ptr[u] += 1
                
                if v == p_parent[u]:
                    continue
                
                if p_ids[v] != -1:
                    # Back edge
                    if p_ids[v] < p_low[u]:
                        p_low[u] = p_ids[v]
                else:
                    # Tree edge
                    p_parent[v] = u
                    p_ids[v] = p_low[v] = timer
                    timer += 1
                    p_stack[stack_top] = v
                    stack_top += 1
                    if u == root:
                        root_children += 1
            else:
                # Finished processing u
                stack_top -= 1
                if p_parent[u] != -1:
                    p = p_parent[u]
                    if p_low[u] < p_low[p]:
                        p_low[p] = p_low[u]
                    if p_low[u] >= p_ids[p] and p != root:
                        p_is_ap[p] = 1
        
        if root_children > 1:
            p_is_ap[root] = 1

    for i in range(num_nodes):
        if p_is_ap[i]:
            ap_list.push_back(i)

    free(p_ids)
    free(p_low)
    free(p_parent)
    free(p_current_edge_ptr)
    free(p_stack)
    free(p_is_ap)
    
    return ap_list
    
    return ap_list