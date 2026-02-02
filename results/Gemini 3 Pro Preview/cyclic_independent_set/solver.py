import numpy as np
from numba import njit

@njit
def greedy_numba(sorted_indices, children, blocked, deltas, num_nodes, strides):
    n = children.shape[1]
    num_deltas = deltas.shape[0]
    solution_indices = []
    
    for idx in sorted_indices:
        if blocked[idx]:
            continue
        
        solution_indices.append(idx)
        
        # Get coordinates of current node
        u = children[idx]
        
        for i in range(num_deltas):
            nbr_idx = 0
            for k in range(n):
                val = u[k] + deltas[i, k]
                if val == num_nodes:
                    val = 0
                elif val == -1:
                    val = num_nodes - 1
                nbr_idx += val * strides[k]
            
            blocked[nbr_idx] = True
            
    return solution_indices

class Solver:
    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem
        
        # Constants
        mod = num_nodes - 2
        if mod <= 0: mod = 1
        
        # Precompute powers for modulo arithmetic
        powers_mod = np.array([pow(num_nodes, n - 1 - k, mod) for k in range(n)], dtype=np.int32)
        
        # Compute distribution of C = sum((1 + v_k)*7^(n-1-k)) mod 5
        # v_k in {2, 4, ..., 2(n-1)}
        
        dist = np.zeros(mod, dtype=np.float64)
        dist[0] = 1.0
        
        js = np.arange(1, n, dtype=np.int32)
        if len(js) > 0:
            base_vals = (1 + 2 * js)
            for k in range(n):
                p = powers_mod[k]
                vals = (base_vals * p) % mod
                
                current_dist = np.zeros(mod, dtype=np.float64)
                for v in vals:
                    current_dist[v] += 1
                
                new_dist = np.zeros(mod, dtype=np.float64)
                for r1 in range(mod):
                    if dist[r1] == 0: continue
                    for r2 in range(mod):
                        if current_dist[r2] == 0: continue
                        new_dist[(r1 + r2) % mod] += dist[r1] * current_dist[r2]
                dist = new_dist

        # Generate children in lexicographical order
        grids = np.indices((num_nodes,) * n, dtype=np.int16)
        children = grids.reshape(n, -1).T.astype(np.int32)
        
        # Compute D = sum( clipped(child[k]) * 7^(n-1-k) ) mod 5
        clipped = np.minimum(children, num_nodes - 3)
        D = np.dot(clipped, powers_mod) % mod
        
        # Map D to Score
        score_map = np.zeros(mod, dtype=np.float64)
        for d_val in range(mod):
            s = 0.0
            for r in range(mod):
                s += dist[r] * ((r + d_val) % mod)
            score_map[d_val] = s
            
        scores = score_map[D]
        
        # Sort indices descending by score, stable to preserve lexicographical order for ties
        sorted_indices = np.argsort(-scores, kind='stable').astype(np.int32)
        
        # Prepare for greedy
        blocked = np.zeros(len(children), dtype=bool)
        
        # Generate deltas
        d_grids = np.indices((3,) * n, dtype=np.int8)
        deltas = d_grids.reshape(n, -1).T - 1
        # Filter out zero delta
        deltas = deltas[np.any(deltas != 0, axis=1)]
        deltas = np.ascontiguousarray(deltas)
        
        strides = np.array([num_nodes**i for i in range(n-1, -1, -1)], dtype=np.int32)
        
        final_indices = greedy_numba(sorted_indices, children, blocked, deltas, num_nodes, strides)
        
        res = [tuple(children[i]) for i in final_indices]
        return res