import itertools
import numpy as np

class Solver:
    # Cache is not used in this version, but the class structure is kept.
    _precomp_cache = {}

    def _priority_reference(self, v, n, num_nodes):
        """
        A direct copy of the reference implementation's priority function,
        with a hypothesized fix in the loop range.
        """
        M_prime = num_nodes - 2
        if M_prime <= 0:
            return 0

        mult_prime = [(num_nodes**i) % M_prime for i in range(n - 1, -1, -1)]
        v_clipped = [min(c, num_nodes - 3) for c in v]
        K_prime = sum((1 + v_clipped[i]) * mult_prime[i] for i in range(n)) % M_prime

        score = 0
        if n > 1:
            # HYPOTHESIS: The range should be (0, ..., n-1) not (1, ..., n-1).
            for V in itertools.product(range(n), repeat=n):
                N_prime = sum(V[i] * mult_prime[i] for i in range(n)) % M_prime
                T = (2 * N_prime) % M_prime
                score += (K_prime + T) % M_prime
        
        return score

    def solve(self, problem, **kwargs):
        """
        Computes an optimal independent set for the n-th strong product of a cyclic graph.
        """
        num_nodes, n = problem

        if n == 0:
            return [()]

        # --- Score Calculation (using slow, correct reference implementation) ---
        children = list(itertools.product(range(num_nodes), repeat=n))
        scores = [self._priority_reference(v, n, num_nodes) for v in children]

        # --- Greedy Selection (Pure Python/NumPy) ---
        num_candidates = len(children)
        available = np.ones(num_candidates, dtype=np.bool_)
        selected_indices_list = []
        
        # Use a dictionary for fast lookups of vertex coordinates to their index.
        child_to_idx = {child: i for i, child in enumerate(children)}
        
        to_block = list(itertools.product([-1, 0, 1], repeat=n))
        
        # Use np.array for argsort.
        sorted_indices = np.argsort(np.array(scores), kind='stable')[::-1]

        for i in sorted_indices:
            if available[i]:
                selected_indices_list.append(i)
                v = children[i]
                
                # Block neighbors
                for shift in to_block:
                    neighbor = tuple(np.mod(np.array(v) + shift, num_nodes))
                    neighbor_idx = child_to_idx.get(neighbor)
                    if neighbor_idx is not None:
                        available[neighbor_idx] = False

        return [children[i] for i in selected_indices_list]