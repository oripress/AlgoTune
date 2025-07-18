import itertools
import numba
import numpy as np

@numba.njit
def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    """
    Numba-accelerated greedy algorithm for finding an independent set.
    """
    n_candidates = len(children)
    n = len(children[0])
    blocked = np.zeros(n_candidates, dtype=np.bool_)
    selected_indices = []
    
    while True:
        # Find the candidate with the highest score that is not blocked
        best_idx = -1
        best_score = -np.inf
        
        for i in range(n_candidates):
            if not blocked[i] and scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        
        if best_idx == -1:
            break
            
        # Select this candidate
        selected_indices.append(best_idx)
        selected = children[best_idx]
        
        # Block all conflicting candidates
        for shift in to_block:
            # Compute neighbor with cyclic wrapping
            neighbor = np.empty(n, dtype=np.int32)
            for j in range(n):
                neighbor[j] = (selected[j] + shift[j]) % num_nodes
            
            # Convert neighbor to index
            idx = 0
            for j in range(n):
                idx += neighbor[j] * powers[j]
            
            # Block this neighbor
            blocked[idx] = True
    
    return selected_indices

class Solver:
    def _priority(self, el, num_nodes, n):
        """
        Compute the priority for a candidate node.
        """
        el_clipped = np.clip(el, a_min=None, a_max=num_nodes - 3)
        values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)))
        multipliers = np.array([num_nodes**i for i in range(n - 1, -1, -1)], dtype=np.int32)
        x = np.sum((1 + values + el_clipped) * multipliers, axis=-1)
        return np.sum(x % (num_nodes - 2), dtype=float)
    
    def solve(self, problem, **kwargs):
        """
        Solve the cyclic graph independent set problem.
        """
        num_nodes, n = problem
        
        # Precompute all candidate vertices
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        # Compute initial scores for all candidates
        scores = np.array([self._priority(tuple(child), num_nodes, n) for child in children])
        # All possible shifts used for blocking
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        # Precompute powers for index conversion
        powers = num_nodes ** np.arange(n - 1, -1, -1)
        
        # Call the accelerated numba solver
        selected_indices = solve_independent_set_numba(
            children, scores, to_block, powers, num_nodes
        )
        
        # Return the selected candidates as a list of tuples
        return [tuple(children[i]) for i in selected_indices]