from typing import Any
import numpy as np
from numba import njit

@njit(cache=True)
def find_aps_numba(num_nodes, adj_data, adj_indptr):
    if num_nodes == 0:
        return np.zeros(0, dtype=np.int32)
    
    disc = np.full(num_nodes, -1, dtype=np.int32)
    low = np.zeros(num_nodes, dtype=np.int32)
    parent = np.full(num_nodes, -1, dtype=np.int32)
    is_ap = np.zeros(num_nodes, dtype=np.bool_)
    
    stack_node = np.empty(num_nodes, dtype=np.int32)
    stack_idx = np.empty(num_nodes, dtype=np.int32)
    
    timer = 0
    
    for root in range(num_nodes):
        if disc[root] != -1:
            continue
        
        disc[root] = timer
        low[root] = timer
        timer += 1
        root_children = 0
        
        sp = 1
        stack_node[0] = root
        stack_idx[0] = adj_indptr[root]
        
        while sp > 0:
            u = stack_node[sp - 1]
            idx = stack_idx[sp - 1]
            adj_end = adj_indptr[u + 1]
            
            if idx < adj_end:
                v = adj_data[idx]
                stack_idx[sp - 1] = idx + 1
                
                if disc[v] == -1:
                    parent[v] = u
                    disc[v] = timer
                    low[v] = timer
                    timer += 1
                    stack_node[sp] = v
                    stack_idx[sp] = adj_indptr[v]
                    sp += 1
                    if u == root:
                        root_children += 1
                elif v != parent[u]:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            else:
                sp -= 1
                p = parent[u]
                if p != -1:
                    if low[u] < low[p]:
                        low[p] = low[u]
                    if p != root and low[u] >= disc[p]:
                        is_ap[p] = True
        
        if root_children > 1:
            is_ap[root] = True
    
    # Count and extract articulation points
    count = 0
    for i in range(num_nodes):
        if is_ap[i]:
            count += 1
    
    result = np.empty(count, dtype=np.int32)
    j = 0
    for i in range(num_nodes):
        if is_ap[i]:
            result[j] = i
            j += 1
    
    return result

@njit(cache=True)
def build_csr(num_nodes, edges_u, edges_v):
    n_edges = len(edges_u)
    
    degree = np.zeros(num_nodes, dtype=np.int32)
    for i in range(n_edges):
        degree[edges_u[i]] += 1
        degree[edges_v[i]] += 1
    
    adj_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(num_nodes):
        adj_indptr[i + 1] = adj_indptr[i] + degree[i]
    
    adj_data = np.empty(2 * n_edges, dtype=np.int32)
    adj_ptr = adj_indptr[:-1].copy()
    
    for i in range(n_edges):
        u, v = edges_u[i], edges_v[i]
        adj_data[adj_ptr[u]] = v
        adj_ptr[u] += 1
        adj_data[adj_ptr[v]] = u
        adj_ptr[v] += 1
    
    return adj_data, adj_indptr

class Solver:
    def __init__(self):
        # Warm up JIT
        dummy_u = np.array([0], dtype=np.int32)
        dummy_v = np.array([1], dtype=np.int32)
        adj_data, adj_indptr = build_csr(2, dummy_u, dummy_v)
        find_aps_numba(2, adj_data, adj_indptr)
    
    def solve(self, problem, **kwargs) -> Any:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        n_edges = len(edges)
        if num_nodes == 0 or n_edges == 0:
            return {"articulation_points": []}
        
        edges_arr = np.array(edges, dtype=np.int32)
        edges_u = np.ascontiguousarray(edges_arr[:, 0])
        edges_v = np.ascontiguousarray(edges_arr[:, 1])
        
        adj_data, adj_indptr = build_csr(num_nodes, edges_u, edges_v)
        result = find_aps_numba(num_nodes, adj_data, adj_indptr)
        
        return {"articulation_points": result.tolist()}