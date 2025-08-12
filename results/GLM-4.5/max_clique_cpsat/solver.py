import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> list[int]:
        """
        Solves the maximum clique problem using a simple branch and bound algorithm.
        
        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of node indices that form a maximum clique.
        """
        n = len(problem)
        adj_matrix = np.array(problem, dtype=np.bool_)
        
        # Precompute adjacency lists for fast lookup
        adj_lists = []
        for i in range(n):
            adj_lists.append(set(np.where(adj_matrix[i])[0]))
        
        best_clique = []
        
        # Simple branch and bound
        def search(current_clique, candidates):
            nonlocal best_clique
            
            if not candidates:
                if len(current_clique) > len(best_clique):
                    best_clique = current_clique.copy()
                return
            
            # Prune if we can't beat the best
            if len(current_clique) + len(candidates) <= len(best_clique):
                return
            
            # Try adding each candidate
            for node in list(candidates):
                # Add node to clique
                current_clique.append(node)
                
                # New candidates are neighbors of node that are still in candidates
                new_candidates = candidates & adj_lists[node]
                
                # Recurse
                search(current_clique, new_candidates)
                
                # Backtrack
                current_clique.pop()
                candidates.remove(node)
        
        # Start with all nodes as candidates
        search([], set(range(n)))
        
        return sorted(best_clique)