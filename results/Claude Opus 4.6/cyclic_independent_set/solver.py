import itertools
import numpy as np
import numba
from typing import Any

@numba.njit(cache=True)
def greedy_select(children, scores, to_block, powers, num_nodes):
    n_candidates = children.shape[0]
    n_dims = children.shape[1]
    n_blocks = to_block.shape[0]
    blocked = np.zeros(n_candidates, dtype=numba.boolean)
    
    order = np.argsort(-scores)
    
    selected = np.empty(n_candidates, dtype=np.int64)
    count = 0
    for idx in range(n_candidates):
        i = order[idx]
        if blocked[i]:
            continue
        selected[count] = i
        count += 1
        for j in range(n_blocks):
            index = np.int64(0)
            for k in range(n_dims):
                coord = (children[i, k] + to_block[j, k]) % num_nodes
                index += coord * powers[k]
            blocked[index] = True
    
    return selected[:count]

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        num_nodes, n = problem
        
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        
        scores = self._compute_all_priorities(children, num_nodes, n)
        
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        
        powers = num_nodes ** np.arange(n - 1, -1, -1).astype(np.int64)
        
        selected_indices = greedy_select(children, scores, to_block, powers, num_nodes)
        
        return [tuple(int(x) for x in children[i]) for i in selected_indices]
    
    def _compute_all_priorities(self, children, num_nodes, n):
        if n <= 1:
            return np.zeros(len(children), dtype=np.float64)
        
        values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)), dtype=np.int64)
        multipliers = np.array([num_nodes**i for i in range(n - 1, -1, -1)], dtype=np.int64)
        
        children_clipped = np.clip(children, a_min=None, a_max=num_nodes - 3).astype(np.int64)
        
        mod = num_nodes - 2
        
        base_x = np.sum((1 + values) * multipliers, axis=1)
        child_contrib = children_clipped @ multipliers
        
        scores = np.zeros(len(children), dtype=np.float64)
        
        for j in range(len(base_x)):
            x = base_x[j] + child_contrib
            scores += (x % mod).astype(np.float64)
        
        return scores