import numpy as np
from numba import njit
from typing import Any

@njit(cache=True)
def backtrack_solve(adj1_flat, adj1_ptr, adj2_flat, adj2_ptr, cands_flat, cands_ptr, n):
    mapping = np.full(n, -1, dtype=np.int32)
    used = np.zeros(n, dtype=np.bool_)
    
    # Order nodes by candidate count (most constrained first), break ties by degree
    cand_counts = np.array([cands_ptr[i+1] - cands_ptr[i] for i in range(n)], dtype=np.int32)
    deg = np.array([adj1_ptr[i+1] - adj1_ptr[i] for i in range(n)], dtype=np.int32)
    order = np.argsort(cand_counts * 10000 - deg)
    
    # Stack: (node_idx, cand_offset)
    stack = np.zeros((n + 1, 2), dtype=np.int32)
    stack_ptr = 0
    stack[stack_ptr, 0] = 0
    stack[stack_ptr, 1] = 0
    stack_ptr += 1
    
    while stack_ptr > 0:
        stack_ptr -= 1
        idx = stack[stack_ptr, 0]
        cand_offset = stack[stack_ptr, 1]
        
        if idx == n:
            return mapping
        
        u = order[idx]
        
        # Undo previous assignment if backtracking
        if mapping[u] != -1:
            used[mapping[u]] = False
            mapping[u] = -1
        
        start = cands_ptr[u]
        end = cands_ptr[u + 1]
        
        found = False
        for ci in range(cand_offset, end - start):
            v = cands_flat[start + ci]
            if used[v]:
                continue
            
            # Check edge constraints
            valid = True
            for prev_idx in range(idx):
                u_prev = order[prev_idx]
                v_prev = mapping[u_prev]
                
                # Check edge in G1
                edge1 = False
                for k in range(adj1_ptr[u], adj1_ptr[u + 1]):
                    if adj1_flat[k] == u_prev:
                        edge1 = True
                        break
                
                # Check edge in G2
                edge2 = False
                for k in range(adj2_ptr[v], adj2_ptr[v + 1]):
                    if adj2_flat[k] == v_prev:
                        edge2 = True
                        break
                
                if edge1 != edge2:
                    valid = False
                    break
            
            if valid:
                mapping[u] = v
                used[v] = True
                # Push backtrack state then forward state
                stack[stack_ptr, 0] = idx
                stack[stack_ptr, 1] = ci + 1
                stack_ptr += 1
                stack[stack_ptr, 0] = idx + 1
                stack[stack_ptr, 1] = 0
                stack_ptr += 1
                found = True
                break
        
        if not found and idx > 0:
            # Backtracking will happen naturally via stack
            pass
    
    return mapping

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        n = problem["num_nodes"]
        edges_g1 = problem["edges_g1"]
        edges_g2 = problem["edges_g2"]
        
        if n == 0:
            return {"mapping": []}
        
        # Build adjacency lists
        adj1 = [[] for _ in range(n)]
        adj2 = [[] for _ in range(n)]
        
        for u, v in edges_g1:
            adj1[u].append(v)
            adj1[v].append(u)
        
        for u, v in edges_g2:
            adj2[u].append(v)
            adj2[v].append(u)
        
        # WL coloring
        colors1 = [len(adj1[i]) for i in range(n)]
        colors2 = [len(adj2[i]) for i in range(n)]
        
        for _ in range(3):
            new_colors1 = [hash((colors1[i], tuple(sorted(colors1[j] for j in adj1[i])))) for i in range(n)]
            new_colors2 = [hash((colors2[i], tuple(sorted(colors2[j] for j in adj2[i])))) for i in range(n)]
            colors1 = new_colors1
            colors2 = new_colors2
        
        # Build candidates
        color_to_nodes2 = {}
        for v in range(n):
            c = colors2[v]
            if c not in color_to_nodes2:
                color_to_nodes2[c] = []
            color_to_nodes2[c].append(v)
        
        candidates = [color_to_nodes2.get(colors1[u], []) for u in range(n)]
        
        # Flatten for numba
        cands_flat = []
        cands_ptr = [0]
        for c in candidates:
            cands_flat.extend(c)
            cands_ptr.append(len(cands_flat))
        
        adj1_flat = []
        adj1_ptr = [0]
        for a in adj1:
            adj1_flat.extend(a)
            adj1_ptr.append(len(adj1_flat))
        
        adj2_flat = []
        adj2_ptr = [0]
        for a in adj2:
            adj2_flat.extend(a)
            adj2_ptr.append(len(adj2_flat))
        
        result = backtrack_solve(
            np.array(adj1_flat, dtype=np.int32),
            np.array(adj1_ptr, dtype=np.int32),
            np.array(adj2_flat, dtype=np.int32),
            np.array(adj2_ptr, dtype=np.int32),
            np.array(cands_flat, dtype=np.int32),
            np.array(cands_ptr, dtype=np.int32),
            n
        )
        
        return {"mapping": result.tolist()}