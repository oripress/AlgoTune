import numpy as np
from typing import List, Set, Any
import numba
from numba import jit, int32
from numba.typed import List as NumbaList

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Solves the maximum clique problem using an optimized algorithm.
        
        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of node indices that form a maximum clique.
        """
        n = len(problem)
        if n == 0:
            return []
        
        # Convert to numpy array for faster access
        adj_matrix = np.array(problem, dtype=np.int32)
        
        # Find maximum clique using optimized branch and bound
        max_clique = self._find_maximum_clique(adj_matrix, n)
        
        return list(max_clique)
    
    def _find_maximum_clique(self, adj_matrix: np.ndarray, n: int) -> List[int]:
        """
        Find maximum clique using branch and bound with pruning.
        """
        # Precompute degrees for ordering
        degrees = np.sum(adj_matrix, axis=0)
        
        # Order vertices by degree (descending) for better pruning
        vertex_order = np.argsort(-degrees)
        
        # Reorder adjacency matrix
        reordered_adj = adj_matrix[vertex_order][:, vertex_order]
        
        # Use branch and bound
        max_clique = []
        candidates = list(range(n))
        current = []
        
        def branch_and_bound(candidates, current):
            nonlocal max_clique
            
            if not candidates:
                if len(current) > len(max_clique):
                    max_clique = current.copy()
                return
            
            # Pruning: if current + all candidates can't beat max, skip
            if len(current) + len(candidates) <= len(max_clique):
                return
            
            # Try each candidate
            for i, v in enumerate(candidates):
                # Create new candidate list with only neighbors of v
                new_candidates = []
                for u in candidates[i+1:]:
                    if reordered_adj[v][u] == 1:
                        new_candidates.append(u)
                
                current.append(v)
                branch_and_bound(new_candidates, current)
                current.pop()
        
        branch_and_bound(candidates, current)
        
        # Map back to original indices
        return [vertex_order[i] for i in max_clique]