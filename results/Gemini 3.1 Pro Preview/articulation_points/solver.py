import numpy as np
from numba import njit
from typing import Any

@njit
def find_articulation_points(num_nodes, edges_u, edges_v):
    num_edges = len(edges_u)
    degree = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_edges):
        degree[edges_u[i]] += 1
        degree[edges_v[i]] += 1
        
    head = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(num_nodes):
        head[i + 1] = head[i] + degree[i]
        
    adj = np.zeros(head[num_nodes], dtype=np.int32)
    cur_head = head[:-1].copy()
    for i in range(num_edges):
        u = edges_u[i]
        v = edges_v[i]
        adj[cur_head[u]] = v
        cur_head[u] += 1
        adj[cur_head[v]] = u
        cur_head[v] += 1
        
    visited = np.zeros(num_nodes, dtype=np.bool_)
    tin = np.zeros(num_nodes, dtype=np.int32)
    low = np.zeros(num_nodes, dtype=np.int32)
    timer = 0
    timer = 0
    
    is_ap = np.zeros(num_nodes, dtype=np.bool_)
    parent = np.full(num_nodes, -1, dtype=np.int32)
    
    stack_u = np.zeros(num_nodes, dtype=np.int32)
    stack_p = np.zeros(num_nodes, dtype=np.int32)
    stack_edge_idx = np.zeros(num_nodes, dtype=np.int32)
    stack_children = np.zeros(num_nodes, dtype=np.int32)
    
    for i in range(num_nodes):
        if not visited[i]:
            top = 0
            stack_u[top] = i
            stack_p[top] = -1
            stack_edge_idx[top] = head[i]
            stack_children[top] = 0
            parent[i] = -1
            
            visited[i] = True
            tin[i] = low[i] = timer
            timer += 1
            
            while top >= 0:
                u = stack_u[top]
                p = stack_p[top]
                edge_idx = stack_edge_idx[top]
                
                if edge_idx < head[u + 1]:
                    v = adj[edge_idx]
                    stack_edge_idx[top] += 1
                    
                    if v == p:
                        continue
                    if visited[v]:
                        low[u] = min(low[u], tin[v])
                    else:
                        parent[v] = u
                        visited[v] = True
                        tin[v] = low[v] = timer
                        timer += 1
                        stack_children[top] += 1
                        
                        top += 1
                        stack_u[top] = v
                        stack_p[top] = u
                        stack_edge_idx[top] = head[v]
                        stack_children[top] = 0
                else:
                    if p != -1:
                        low[p] = min(low[p], low[u])
                        if parent[p] != -1:
                            if low[u] >= tin[p]:
                                is_ap[p] = True
                    else:
                        if stack_children[top] > 1:
                            is_ap[u] = True
                    top -= 1
    ap_count = 0
    for i in range(num_nodes):
        if is_ap[i]:
            ap_count += 1
    res = np.zeros(ap_count, dtype=np.int32)
    idx = 0
    for i in range(num_nodes):
        if is_ap[i]:
            res[idx] = i
            idx += 1
    return res

class Solver:
    def __init__(self):
        dummy_edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        find_articulation_points(3, dummy_edges[:, 0], dummy_edges[:, 1])

    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        if len(edges) == 0:
            return {"articulation_points": []}
            
        edges_arr = np.array(edges, dtype=np.int32)
        edges_u = edges_arr[:, 0]
        edges_v = edges_arr[:, 1]
        
        ap_arr = find_articulation_points(num_nodes, edges_u, edges_v)
        return {"articulation_points": ap_arr.tolist()}