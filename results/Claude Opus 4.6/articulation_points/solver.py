import numpy as np
from typing import Any
from numba import njit

@njit(cache=True)
def build_and_find(num_nodes, src, dst):
    """Build CSR adjacency and find articulation points in one numba call."""
    num_edges = len(src)
    
    # Count degrees
    degree = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_edges):
        degree[src[i]] += 1
        degree[dst[i]] += 1
    
    # Build offsets
    adj_offsets = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(num_nodes):
        adj_offsets[i + 1] = adj_offsets[i] + degree[i]
    
    # Build flat adjacency list
    total = adj_offsets[num_nodes]
    adj_flat = np.empty(total, dtype=np.int32)
    pos = np.empty(num_nodes, dtype=np.int32)
    for i in range(num_nodes):
        pos[i] = adj_offsets[i]
    
    for i in range(num_edges):
        u = src[i]
        v = dst[i]
        adj_flat[pos[u]] = v
        pos[u] += 1
        adj_flat[pos[v]] = u
        pos[v] += 1
    
    # Now find articulation points using Tarjan's iterative algorithm
    disc = np.full(num_nodes, -1, dtype=np.int32)
    low = np.full(num_nodes, -1, dtype=np.int32)
    is_ap = np.zeros(num_nodes, dtype=np.bool_)
    
    timer = 0
    
    stack_node = np.empty(num_nodes, dtype=np.int32)
    stack_parent = np.empty(num_nodes, dtype=np.int32)
    stack_edge_idx = np.empty(num_nodes, dtype=np.int32)
    stack_child_count = np.empty(num_nodes, dtype=np.int32)
    
    for start in range(num_nodes):
        if disc[start] != -1:
            continue
        
        sp = 0
        disc[start] = timer
        low[start] = timer
        timer += 1
        stack_node[sp] = start
        stack_parent[sp] = -1
        stack_edge_idx[sp] = adj_offsets[start]
        stack_child_count[sp] = 0
        
        while sp >= 0:
            u = stack_node[sp]
            parent = stack_parent[sp]
            edge_idx = stack_edge_idx[sp]
            end_idx = adj_offsets[u + 1]
            
            if edge_idx < end_idx:
                v = adj_flat[edge_idx]
                stack_edge_idx[sp] = edge_idx + 1
                
                if v == parent:
                    continue
                
                if disc[v] != -1:
                    if low[u] > disc[v]:
                        low[u] = disc[v]
                else:
                    disc[v] = timer
                    low[v] = timer
                    timer += 1
                    sp += 1
                    stack_node[sp] = v
                    stack_parent[sp] = u
                    stack_edge_idx[sp] = adj_offsets[v]
                    stack_child_count[sp] = 0
            else:
                sp -= 1
                if sp >= 0:
                    p = stack_node[sp]
                    p_parent = stack_parent[sp]
                    if low[p] > low[u]:
                        low[p] = low[u]
                    stack_child_count[sp] += 1
                    
                    if p_parent == -1:
                        if stack_child_count[sp] > 1:
                            is_ap[p] = True
                    else:
                        if low[u] >= disc[p]:
                            is_ap[p] = True
    
    # Collect results
    count = 0
    for i in range(num_nodes):
        if is_ap[i]:
            count += 1
    result = np.empty(count, dtype=np.int32)
    idx = 0
    for i in range(num_nodes):
        if is_ap[i]:
            result[idx] = i
            idx += 1
    return result

class Solver:
    def __init__(self):
        # Warm up numba
        src = np.array([0], dtype=np.int32)
        dst = np.array([1], dtype=np.int32)
        build_and_find(2, src, dst)
    
    def solve(self, problem, **kwargs) -> Any:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        if num_nodes == 0 or len(edges) == 0:
            return {"articulation_points": []}
        
        edges_arr = np.array(edges, dtype=np.int32)
        src = edges_arr[:, 0]
        dst = edges_arr[:, 1]
        
        result = build_and_find(num_nodes, src, dst)
        
        return {"articulation_points": result.tolist()}