import itertools
import numba
import numpy as np

class Solver:
    def solve(self, problem: tuple[int, int]) -> list[tuple[int, ...]]:
        """
        Solve the cyclic graph independent set problem.
        """
        num_nodes, n = problem

        # Precompute all candidate vertices.
        children = np.array(list(itertools.product(range(num_nodes), repeat=n)), dtype=np.int32)
        
        # Precompute values and multipliers once
        values = 2 * np.array(list(itertools.product(range(1, n), repeat=n)), dtype=np.int32)
        multipliers = np.array([num_nodes**i for i in range(n - 1, -1, -1)], dtype=np.int32)
        
        # Vectorized score computation
        children_clipped = np.minimum(children, num_nodes - 3)
        child_contrib = children_clipped @ multipliers  # (num_candidates,)
        values_contrib = values @ multipliers  # (num_values,)
        base = multipliers.sum()
        
        # Broadcast to compute all scores at once
        # (num_candidates, 1) + (num_values,) -> (num_candidates, num_values)
        all_sums = (base + child_contrib[:, None] + values_contrib[None, :]) % (num_nodes - 2)
        scores = all_sums.sum(axis=1).astype(np.float64)  # (num_candidates,)
        
        # All possible shifts used for blocking.
        to_block = np.array(list(itertools.product([-1, 0, 1], repeat=n)), dtype=np.int32)
        # Precompute powers for index conversion.
        powers = num_nodes ** np.arange(n - 1, -1, -1)

        # Call the optimized numba solver
        selected_indices = solve_independent_set_numba(
            children, scores, to_block, powers, num_nodes
        )

        # Return the selected candidates as a list of tuples.
        return [tuple(children[i]) for i in selected_indices]

@numba.njit(fastmath=True)
def compute_scores_numba(children, values, multipliers, num_nodes):
    """Compute scores for all children using numba with fastmath."""
    num_candidates = children.shape[0]
    n = children.shape[1]
    num_values = values.shape[0]
    scores = np.zeros(num_candidates, dtype=np.float64)
    
    for i in range(num_candidates):
        child = children[i]
        child_clipped = np.minimum(child, num_nodes - 3)
        
        score = 0.0
        for j in range(num_values):
            x_val = 0
            for k in range(n):
                x_val += (1 + values[j, k] + child_clipped[k]) * multipliers[k]
            score += x_val % (num_nodes - 2)
        scores[i] = score
    
    return scores

@numba.njit(fastmath=True)
def solve_independent_set_numba_optimized(children, values, multipliers, to_block, powers, num_nodes):
    """
    Greedy algorithm to find an independent set, fully optimized with numba and fastmath.
    """
    n = children.shape[1]
    num_candidates = children.shape[0]
    
    # Compute scores
    scores = compute_scores_numba(children, values, multipliers, num_nodes)
    
    # Track which candidates are still available
    available = np.ones(num_candidates, dtype=np.bool_)
    selected = np.empty(num_candidates, dtype=np.int64)
    count = 0
    
    while True:
        # Find the highest scoring available candidate
        best_idx = -1
        best_score = -np.inf
        
        for i in range(num_candidates):
            if available[i] and scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        
        if best_idx == -1:
            break
        
        # Select this candidate
        selected[count] = best_idx
        count += 1
        available[best_idx] = False
        
        # Block all conflicting candidates
        current = children[best_idx]
        
        for shift_idx in range(to_block.shape[0]):
            shift = to_block[shift_idx]
            blocked = (current + shift) % num_nodes
            
            # Convert to index
            blocked_idx = 0
            for j in range(n):
                blocked_idx += blocked[j] * powers[j]
            
            if 0 <= blocked_idx < num_candidates:
                available[blocked_idx] = False
    
    return selected[:count]
    
    return selected[:count]

@numba.njit
def solve_independent_set_numba(children, scores, to_block, powers, num_nodes):
    """
    Greedy algorithm to find an independent set, accelerated with numba.
    """
    n = children.shape[1]
    num_candidates = children.shape[0]
    
    # Track which candidates are still available
    available = np.ones(num_candidates, dtype=np.bool_)
    selected = np.empty(num_candidates, dtype=np.int64)
    count = 0
    
    while True:
        # Find the highest scoring available candidate
        best_idx = -1
        best_score = -np.inf
        
        for i in range(num_candidates):
            if available[i] and scores[i] > best_score:
                best_score = scores[i]
                best_idx = i
        
        if best_idx == -1:
            break
        
        # Select this candidate
        selected[count] = best_idx
        count += 1
        available[best_idx] = False
        
        # Block all conflicting candidates
        current = children[best_idx]
        
        for shift in to_block:
            # Compute the blocked vertex
            blocked = (current + shift) % num_nodes
            
            # Convert to index
            idx = 0
            for j in range(n):
                idx += blocked[j] * powers[j]
            
            if 0 <= idx < num_candidates:
                available[idx] = False
    
    return selected[:count]