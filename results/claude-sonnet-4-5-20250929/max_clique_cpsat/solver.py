from numba import njit
import numpy as np

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Solves the maximum clique problem using a custom branch-and-bound algorithm.
        
        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of node indices that form a maximum clique.
        """
        n = len(problem)
        
        # Quick checks
        if n == 0:
            return []
        if n == 1:
            return [0]
        
        # Convert to numpy array for Numba
        adj = np.array(problem, dtype=np.int8)
        
        # Order vertices by degree (greedy heuristic for better pruning)
        degrees = adj.sum(axis=1)
        vertices = np.argsort(-degrees).astype(np.int32)
        
        # Call JIT-compiled function
        max_clique = self._find_max_clique(adj, vertices)
        return sorted(max_clique.tolist())
    
    @staticmethod
    @njit
    def _find_max_clique(adj, vertices):
        """Find maximum clique using branch and bound."""
        n = len(vertices)
        
        # Start with a greedy clique for better initial bound
        max_clique = _greedy_clique(adj, vertices)
        
        current = np.empty(n, dtype=np.int32)
        candidates = vertices.copy()
        
        max_clique = _branch_and_bound_nb(adj, current, 0, candidates, n, max_clique)
        return max_clique

@njit
def _greedy_clique(adj, vertices):
    """Build initial clique greedily."""
    clique = np.empty(len(vertices), dtype=np.int32)
    clique_len = 0
    
    for v in vertices:
        # Check if v is connected to all vertices in clique
        valid = True
        for i in range(clique_len):
            if adj[v, clique[i]] == 0:
                valid = False
                break
        if valid:
            clique[clique_len] = v
            clique_len += 1
    
    return clique[:clique_len].copy()

@njit
def _branch_and_bound_nb(adj, current, current_len, candidates, cand_len, max_clique):
    """Numba-compiled branch and bound."""
    if current_len > len(max_clique):
        max_clique = current[:current_len].copy()
    
    # Prune if we can't possibly improve
    if current_len + cand_len <= len(max_clique):
        return max_clique
    
    for i in range(cand_len):
        v = candidates[i]
        
        # Check if v connects to all vertices in current clique
        valid = True
        for j in range(current_len):
            if adj[v, current[j]] == 0:
                valid = False
                break
        
        if valid:
            # Find neighbors of v in remaining candidates
            new_cand_len = 0
            new_candidates = np.empty(cand_len - i - 1, dtype=np.int32)
            for j in range(i + 1, cand_len):
                if adj[v, candidates[j]] == 1:
                    new_candidates[new_cand_len] = candidates[j]
                    new_cand_len += 1
            
            current[current_len] = v
            max_clique = _branch_and_bound_nb(adj, current, current_len + 1, 
                                              new_candidates, new_cand_len, max_clique)
    
    return max_clique