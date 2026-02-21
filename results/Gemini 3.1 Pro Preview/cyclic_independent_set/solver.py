import itertools
import numba
import numpy as np

@numba.njit
def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    n_children = len(children)
    blocked = np.zeros(n_children, dtype=np.bool_)
    selected_indices = []
    
    while True:
        best_score = -1.0
        best_idx = -1
        for i in range(n_children):
            if not blocked[i]:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_idx = i
                    
        if best_idx == -1:
            break
            
        selected_indices.append(best_idx)
        child = children[best_idx]
        for j in range(len(to_block)):
            shift = to_block[j]
            neighbor_idx = 0
            for k in range(len(child)):
                val = (child[k] + shift[k]) % num_nodes
                neighbor_idx += val * powers[k]
            blocked[neighbor_idx] = True
            
    return selected_indices

class Solver:
    def __init__(self):
        # Precompile numba function
        dummy_children = np.zeros((1, 1), dtype=np.int32)
        dummy_scores = np.zeros(1, dtype=np.float64)
        dummy_to_block = np.zeros((1, 1), dtype=np.int32)
        dummy_powers = np.zeros(1, dtype=np.int32)
        solve_independent_set_numba(dummy_children, dummy_scores, dummy_to_block, dummy_powers, 7)

    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        num_nodes, n = problem
        
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        
        el_clipped = np.clip(children, a_min=None, a_max=num_nodes - 3)
        values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)), dtype=np.int32)
        multipliers = np.array([num_nodes**i for i in range(n - 1, -1, -1)], dtype=np.int32)
        
        base_x = np.sum((1 + values) * multipliers, axis=-1)
        child_x = np.sum(el_clipped * multipliers, axis=-1)
        
        mod_val = num_nodes - 2
        scores = np.sum((base_x[None, :] + child_x[:, None]) % mod_val, axis=1, dtype=np.float64)
        
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        powers = num_nodes ** np.arange(n - 1, -1, -1, dtype=np.int32)
        
        selected_indices = solve_independent_set_numba(
            children, scores, to_block, powers, num_nodes
        )
        
        return [tuple(children[i]) for i in selected_indices]