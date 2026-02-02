import itertools
import numpy as np
from numba import njit

class Solver:
    def solve(self, problem, **kwargs):
        num_nodes, n = problem
        
        # Generate all candidate vertices
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        
        # Compute scores using modular optimization
        scores = self._compute_scores(children, num_nodes, n)
        
        # Generate blocking shifts
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        
        # Powers for index computation
        powers = (num_nodes ** np.arange(n - 1, -1, -1)).astype(np.int64)
        
        # Sort by score (descending) - stable sort preserves original order for ties
        sorted_indices = np.argsort(-scores, kind='stable').astype(np.int64)
        
        # Greedy selection with blocking
        selected_indices = _greedy_select(children, sorted_indices, to_block, powers, num_nodes)
        
        return [tuple(children[i]) for i in selected_indices]
    
    def _compute_scores(self, children, num_nodes, n):
        if n <= 1:
            return np.zeros(len(children), dtype=np.float64)
        
        mod = num_nodes - 2  # 5 for num_nodes=7
        
        # Generate values array: 2 * product(range(1, n), repeat=n)
        values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)), dtype=np.int64)
        
        # Multipliers: [num_nodes^(n-1), ..., num_nodes, 1]
        multipliers = np.array([num_nodes**i for i in range(n - 1, -1, -1)], dtype=np.int64)
        
        # Compute base_x = C1 + V where C1 = sum(multipliers), V[j] = values[j] @ multipliers
        C1 = np.sum(multipliers)
        V = values @ multipliers  # (L,) where L = (n-1)^n
        base_x = C1 + V
        
        # Count residues of base_x mod 'mod'
        # Key optimization: (base_x[j] + C3) % mod = ((base_x[j] % mod) + (C3 % mod)) % mod
        # So we can precompute a lookup table based on C3 % mod
        base_x_mod = base_x.astype(np.int64) % mod
        count = np.bincount(base_x_mod, minlength=mod).astype(np.float64)
        
        # Precompute lookup table: lookup[c] = sum_r count[r] * ((r + c) % mod)
        lookup = np.zeros(mod, dtype=np.float64)
        for c in range(mod):
            for r in range(mod):
                lookup[c] += count[r] * ((r + c) % mod)
        
        # Compute C3 = children_clipped @ multipliers for all children
        children_clipped = np.clip(children, a_min=None, a_max=num_nodes - 3).astype(np.int64)
        C3 = children_clipped @ multipliers  # (N,)
        
        # Get scores via lookup
        c_mod = C3.astype(np.int64) % mod
        return lookup[c_mod]

@njit(cache=True)
def _greedy_select(children, sorted_indices, to_block, powers, num_nodes):
    n_candidates = len(children)
    n = children.shape[1]
    blocked = np.zeros(n_candidates, dtype=np.bool_)
    
    # Preallocate for selected indices
    selected = np.zeros(n_candidates, dtype=np.int64)
    n_selected = 0
    
    for i in range(len(sorted_indices)):
        idx = sorted_indices[i]
        if blocked[idx]:
            continue
        
        selected[n_selected] = idx
        n_selected += 1
        
        # Block all neighbors
        candidate = children[idx]
        for j in range(len(to_block)):
            shift = to_block[j]
            neighbor_idx = 0
            for k in range(n):
                neighbor_k = (candidate[k] + shift[k]) % num_nodes
                neighbor_idx += neighbor_k * powers[k]
            blocked[neighbor_idx] = True
    
    return selected[:n_selected]