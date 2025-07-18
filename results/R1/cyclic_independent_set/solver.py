import itertools
import numpy as np
from numba import njit, int32, float64, boolean
from functools import lru_cache

@njit(float64(int32[:], int32, int32))
def calculate_priority(el, num_nodes, n):
    """Compute priority score for a candidate vertex"""
    el_clipped = np.minimum(el, num_nodes - 3)
    total = 0.0
    for k in range(n):
        term = 1 + 2 * (k + 1) + el_clipped[k]
        total += term * (num_nodes ** (n - k - 1))
    return total % (num_nodes - 2)

@njit
def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    n = children.shape[1]
    num_candidates = children.shape[0]
    num_blockers = to_block.shape[0]
    blocked = np.zeros(num_candidates, dtype=boolean)
    selected = []
    
    while True:
        best_index = -1
        best_score = -np.inf
        for i in range(num_candidates):
            if not blocked[i] and scores[i] > best_score:
                best_score = scores[i]
                best_index = i
        
        if best_index == -1:
            break
        
        selected.append(best_index)
        chosen_child = children[best_index]
        
        for j in range(num_blockers):
            shift = to_block[j]
            shifted_child = (chosen_child + shift) % num_nodes
            index = 0
            for k in range(n):
                index += shifted_child[k] * powers[k]
            if index < num_candidates:
                blocked[index] = True
    
    return np.array(selected, dtype=np.int64)

class Solver:
    def solve(self, problem, **kwargs):
        num_nodes, n = problem
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        
        # Compute scores using vectorized operations
        scores = np.zeros(len(children), dtype=np.float64)
        for i, child in enumerate(children):
            scores[i] = calculate_priority(child, num_nodes, n)
        
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        powers = num_nodes ** np.arange(n-1, -1, -1)
        selected_indices = solve_independent_set_numba(children, scores, to_block, powers, num_nodes)
        return [tuple(children[i]) for i in selected_indices]