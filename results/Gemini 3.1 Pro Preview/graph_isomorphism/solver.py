import numpy as np
from numba import njit
from typing import Any

@njit
def compute_colors(n, adj):
    colors = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        colors[i] = np.sum(adj[i])
        
    for _ in range(3):
        new_colors = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            h = colors[i]
            neighbor_hash = np.uint64(0)
            for j in range(n):
                if adj[i, j]:
                    c = colors[j]
                    c = (c ^ (c >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
                    c = c ^ (c >> np.uint64(27))
                    neighbor_hash += c
            
            new_colors[i] = h * np.uint64(0x9e3779b97f4a7c15) + neighbor_hash
        colors = new_colors
    return colors
@njit
def solve_iso_numba(n, adj1, adj2):
    colors1 = compute_colors(n, adj1)
    colors2 = compute_colors(n, adj2)
    
    freq = np.zeros(n, dtype=np.int32)
    for i in range(n):
        c = colors1[i]
        count = 0
        for j in range(n):
            if colors1[j] == c:
                count += 1
        freq[i] = count
        
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        scores[i] = freq[i] * 10000.0 - np.sum(adj1[i])
        
    order = np.argsort(scores)
    
    mapping = np.full(n, -1, dtype=np.int32)
    inverse_mapping = np.full(n, -1, dtype=np.int32)
    
    cands = np.zeros((n, n), dtype=np.int32)
    cands_len = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        u = order[i]
        idx = 0
        for v in range(n):
            if colors1[u] == colors2[v]:
                cands[i, idx] = v
                idx += 1
        cands_len[i] = idx

    idx = 0
    cand_idx = np.zeros(n, dtype=np.int32)
    
    while idx < n:
        u = order[idx]
        
        found = False
        while cand_idx[idx] < cands_len[idx]:
            v = cands[idx, cand_idx[idx]]
            cand_idx[idx] += 1
            
            if inverse_mapping[v] != -1:
                continue
                
            valid = True
            for i in range(idx):
                prev_u = order[i]
                prev_v = mapping[prev_u]
                if adj1[u, prev_u] != adj2[v, prev_v]:
                    valid = False
                    break
                    
            if valid:
                mapping[u] = v
                inverse_mapping[v] = u
                found = True
                break
                
        if found:
            idx += 1
            if idx < n:
                cand_idx[idx] = 0
        else:
            cand_idx[idx] = 0
            idx -= 1
            if idx < 0:
                break
            u_prev = order[idx]
            v_prev = mapping[u_prev]
            mapping[u_prev] = -1
            inverse_mapping[v_prev] = -1
            
    return mapping

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        n = problem["num_nodes"]
        
        adj1 = np.zeros((n, n), dtype=np.int8)
        for u, v in problem["edges_g1"]:
            adj1[u, v] = 1
            adj1[v, u] = 1
            
        adj2 = np.zeros((n, n), dtype=np.int8)
        for u, v in problem["edges_g2"]:
            adj2[u, v] = 1
            adj2[v, u] = 1
            
        mapping = solve_iso_numba(n, adj1, adj2)
        
        return {"mapping": mapping.tolist()}