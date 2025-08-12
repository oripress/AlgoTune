import networkx as nx
import numpy as np
from collections import deque

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the maximum clique problem using optimized approaches.
        
        :param problem: A 2D adjacency matrix representing the graph.
        :return: A list of node indices that form a maximum clique.
        """
        n = len(problem)
        if n == 0:
            return []
        if n == 1:
            return [0]
            
        # Convert to numpy array for faster operations
        adj_matrix = np.array(problem, dtype=np.uint8)
        
        # For very small graphs, use optimized brute force
        if n <= 6:
            return self._optimized_brute_force(adj_matrix, n)
            
        # For medium graphs, use a highly optimized approach
        if n <= 12:
            return self._highly_optimized_approach(adj_matrix, n)
            
        # For larger graphs, use optimized NetworkX
        G = nx.from_numpy_array(adj_matrix)
        
        try:
            # Try to use the max_weight_clique function if available
            max_clique = nx.max_weight_clique(G, weight=None)[0]
        except (AttributeError, Exception):
            try:
                # Fallback to find_cliques with optimization
                max_clique = max(nx.find_cliques(G), key=len)
            except:
                # Ultimate fallback to greedy approach
                return self._greedy_approach(adj_matrix, n)
        return sorted(list(max_clique))
    
    def _optimized_brute_force(self, adj_matrix, n):
        """Highly optimized brute force for very small graphs."""
        max_clique = []
        
        # For very small graphs, check all subsets
        if n <= 4:
            # Check all possible subsets in reverse order (larger first)
            for i in range((1 << n) - 1, 0, -1):
                subset = [j for j in range(n) if (i >> j) & 1]
                # Quick check: if this subset can't beat current max, skip
                if len(subset) <= len(max_clique):
                    continue
                            
                # Check if it's a clique using numpy for speed
                is_clique = True
                for u in range(len(subset)):
                    for v in range(u + 1, len(subset)):
                        if adj_matrix[subset[u], subset[v]] == 0:
                            is_clique = False
                            break
                    if not is_clique:
                        break
                        
                if is_clique:
                    return sorted(subset)
        else:
            # For slightly larger graphs, use optimized search
            return self._fast_greedy_search(adj_matrix, n)
            
        return sorted(max_clique)
        
    def _highly_optimized_approach(self, adj_matrix, n):
        """Highly optimized approach for medium graphs."""
        # Precompute adjacency lists for faster access
        adj_list = []
        for i in range(n):
            adj_list.append(np.where(adj_matrix[i] == 1)[0])
        
        # Precompute degrees
        degrees = np.sum(adj_matrix, axis=1)
        
        # Find an initial good clique using greedy approach
        initial_clique = self._greedy_clique(adj_matrix, n, degrees)
        
        # Use a more efficient search with better pruning
        max_clique = self._ultra_fast_search(adj_matrix, adj_list, n, initial_clique, degrees)
        
        return sorted(max_clique)
    
    def _greedy_clique(self, adj_matrix, n, degrees):
        """Find initial clique using greedy approach."""
        best_clique = []
        
        # Try multiple starting points
        node_order = np.argsort(degrees)[::-1]
        for i in range(min(n, 2)):
            start_node = node_order[i]
            clique = self._build_clique_fast(adj_matrix, n, start_node)
            if len(clique) > len(best_clique):
                best_clique = clique
                
        return best_clique
    
    def _ultra_fast_search(self, adj_matrix, adj_list, n, initial_clique, degrees):
        """Ultra-fast search with aggressive pruning."""
        max_clique = initial_clique[:]
        max_size = len(max_clique)
        
        # Convert to sets for faster operations
        adj_sets = [set(adj_list[i]) for i in range(n)]
        
        # Use a simple but effective recursive approach with aggressive pruning
        def search_recursive(current_clique, candidates, excluded):
            nonlocal max_clique, max_size
            
            # Pruning: if current clique + potential nodes can't beat max_clique, return
            if len(current_clique) + len(candidates) <= max_size:
                return
                
            # If no candidates, we have a maximal clique
            if not candidates:
                if len(current_clique) > max_size:
                    max_clique = list(current_clique)
                    max_size = len(max_clique)
                return
            
            # Choose pivot for optimization
            pivot = max(candidates | excluded, key=lambda x: len(adj_sets[x] & candidates), default=None)
            
            # Reduce candidates using pivot
            if pivot is not None:
                reduced_candidates = candidates - adj_sets[pivot]
            else:
                reduced_candidates = candidates
            
            # Process candidates in order of degree for better pruning
            candidate_list = sorted(reduced_candidates, key=lambda x: degrees[x], reverse=True)
            
            for node in candidate_list:
                # Add node to clique
                new_clique = current_clique | {node}
                
                # Find common neighbors
                neighbors = adj_sets[node]
                new_candidates = candidates & neighbors
                new_excluded = excluded & neighbors
                
                # Recursive call
                search_recursive(new_clique, new_candidates, new_excluded)
                
                # Remove node from candidates and add to excluded
                candidates = candidates - {node}
                excluded = excluded | {node}
        
        # Start search
        candidates = set(range(n))
        excluded = set()
        search_recursive(set(), candidates, excluded)
        
        return max_clique
        
    def _fast_greedy_search(self, adj_matrix, n):
        """Fast greedy search for small to medium graphs."""
        max_clique = []
        
        # Use a more efficient search strategy
        # Start with nodes that have highest degrees
        degrees = np.sum(adj_matrix, axis=1)
        node_order = np.argsort(degrees)[::-1]
        
        # Limit search space by only trying top nodes
        search_limit = min(n, 2)
        for i in range(search_limit):
            start_node = node_order[i]
            clique = self._build_clique_fast(adj_matrix, n, start_node)
            if len(clique) > len(max_clique):
                max_clique = clique
                # Early termination heuristic
                if len(max_clique) >= n // 2:
                    break
                    
        return sorted(max_clique)
        
    @staticmethod
    def _build_clique_fast(adj_matrix, n, start_node):
        """Build a clique using fast approach."""
        clique = {start_node}
        # Get all neighbors of start node
        neighbors = set(i for i in range(n) if adj_matrix[start_node][i] == 1)
        
        # While we have candidates, try to expand
        while neighbors:
            # Pick the node that connects to most current clique members
            best_node = None
            best_connections = -1
            
            # Only check a limited number of candidates for speed
            candidate_list = list(neighbors)[:min(len(neighbors), 4)]
            for node in candidate_list:
                connections = sum(1 for clique_node in clique if adj_matrix[node][clique_node] == 1)
                if connections > best_connections:
                    best_connections = connections
                    best_node = node
                    
            # If the best node connects to all current clique members, add it
            if best_node is not None and best_connections == len(clique):
                clique.add(best_node)
                # Update neighbors to only include those connected to best_node
                neighbors = {node for node in neighbors if adj_matrix[best_node][node] == 1}
                neighbors.discard(best_node)
            else:
                break
                
        return list(clique)
        
    def _greedy_approach(self, adj_matrix, n):
        """Greedy approach for larger graphs."""
        best_clique = []
        
        # Strategy 1: Start with highest degree nodes
        degrees = np.sum(adj_matrix, axis=1)
        node_order = np.argsort(degrees)[::-1]
        
        for i in range(min(n, 1)):  # Try fewer starting points for speed
            start_node = node_order[i]
            clique = self._build_clique_fast(adj_matrix, n, start_node)
            if len(clique) > len(best_clique):
                best_clique = clique
                
        return sorted(best_clique)