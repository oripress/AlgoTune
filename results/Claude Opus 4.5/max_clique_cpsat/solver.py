import numpy as np
from numba import njit

@njit(cache=True)
def color_and_sort(adj, cands, n_cands, sorted_cands, colors):
    """Color candidates and sort by color ascending."""
    if n_cands == 0:
        return 0
    
    temp_colors = np.zeros(n_cands, dtype=np.int64)
    max_color = 0
    
    for i in range(n_cands):
        v = cands[i]
        used = np.zeros(n_cands + 1, dtype=np.bool_)
        for j in range(i):
            if adj[v, cands[j]]:
                used[temp_colors[j]] = True
        c = 1
        while used[c]:
            c += 1
        temp_colors[i] = c
        if c > max_color:
            max_color = c
    
    # Sort by color using counting sort for stability
    count = np.zeros(max_color + 2, dtype=np.int64)
    for i in range(n_cands):
        count[temp_colors[i] + 1] += 1
    for i in range(1, max_color + 2):
        count[i] += count[i-1]
    
    for i in range(n_cands):
        c = temp_colors[i]
        pos = count[c]
        sorted_cands[pos] = cands[i]
        colors[pos] = c
        count[c] += 1
    
    return max_color

@njit(cache=True)
def find_max_clique_mcq(adj, n):
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    
    best = np.zeros(n, dtype=np.int64)
    best_size = 0
    
    max_depth = n + 1
    stack_clique = np.zeros(n, dtype=np.int64)
    stack_cands = np.zeros((max_depth, n), dtype=np.int64)
    stack_cands_size = np.zeros(max_depth, dtype=np.int64)
    stack_colors = np.zeros((max_depth, n), dtype=np.int64)
    stack_pos = np.zeros(max_depth, dtype=np.int64)
    
    # Order vertices by degree descending
    degrees = np.sum(adj, axis=1)
    order = np.argsort(-degrees)
    init_cands = np.zeros(n, dtype=np.int64)
    for i in range(n):
        init_cands[i] = order[i]
    
    # Color and sort initial candidates
    color_and_sort(adj, init_cands, n, stack_cands[0], stack_colors[0])
    stack_cands_size[0] = n
    stack_pos[0] = n - 1  # Start from last (highest color)
    
    depth = 0
    
    while depth >= 0:
        pos = stack_pos[depth]
        clique_size = depth
        
        # Find next valid candidate (process from highest to lowest index)
        while pos >= 0:
            color = stack_colors[depth, pos]
            if clique_size + color > best_size:
                break
            pos -= 1
        
        if pos < 0:
            # Backtrack
            depth -= 1
            if depth >= 0:
                stack_pos[depth] -= 1
            continue
        
        # Select vertex
        v = stack_cands[depth, pos]
        stack_clique[clique_size] = v
        
        # Build new candidates (all candidates at positions 0..pos-1 adjacent to v)
        new_cands = np.zeros(pos, dtype=np.int64)
        new_size = 0
        for i in range(pos):
            u = stack_cands[depth, i]
            if adj[v, u]:
                new_cands[new_size] = u
                new_size += 1
        
        if new_size == 0:
            # Found a maximal clique
            if clique_size + 1 > best_size:
                best_size = clique_size + 1
                for i in range(clique_size + 1):
                    best[i] = stack_clique[i]
            stack_pos[depth] = pos - 1
        else:
            # Color and sort new candidates
            color_and_sort(adj, new_cands, new_size, stack_cands[depth + 1], stack_colors[depth + 1])
            stack_cands_size[depth + 1] = new_size
            stack_pos[depth] = pos
            stack_pos[depth + 1] = new_size - 1
            depth += 1
    
    return best[:best_size]

class Solver:
    def solve(self, problem, **kwargs):
        n = len(problem)
        if n == 0:
            return []
        
        adj = np.array(problem, dtype=np.int8)
        result = find_max_clique_mcq(adj, n)
        return result.tolist()