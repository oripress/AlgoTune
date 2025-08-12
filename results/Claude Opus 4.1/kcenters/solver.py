import numpy as np
from typing import Any, Dict, List, Set, Tuple
from pysat.solvers import Solver as SATSolver

class Solver:
    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int]) -> Any:
        """
        Solves the k-centers problem using SAT solver with binary search.
        """
        G_dict, k_centers = problem
        
        # Handle edge cases
        if not G_dict or k_centers == 0:
            return set()
        
        # Build graph and compute distances efficiently
        nodes = list(G_dict.keys())
        n = len(nodes)
        
        if k_centers >= n:
            return set(nodes)
        
        # Create node index mapping
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build distance matrix using Floyd-Warshall
        INF = float('inf')
        dist = np.full((n, n), INF, dtype=np.float64)
        
        # Initialize direct edges
        for i in range(n):
            dist[i, i] = 0
            node_i = nodes[i]
            if node_i in G_dict:
                for node_j, weight in G_dict[node_i].items():
                    j = node_to_idx[node_j]
                    dist[i, j] = weight
        
        # Floyd-Warshall algorithm (optimized with NumPy)
        for k in range(n):
            # Broadcast k-th row and column
            dist = np.minimum(dist, dist[:, k:k+1] + dist[k:k+1, :])
        
        # Get unique distances for binary search
        unique_dists = np.unique(dist[dist < INF])
        
        # Greedy heuristic for initial upper bound
        centers_idx = []
        min_dists = np.full(n, INF)
        
        # Choose first center (minimize maximum distance)
        max_dists = np.max(dist, axis=1)
        first_center = np.argmin(max_dists)
        centers_idx.append(first_center)
        min_dists = dist[first_center].copy()
        
        # Choose remaining centers greedily
        for _ in range(1, k_centers):
            farthest = np.argmax(min_dists)
            centers_idx.append(farthest)
            np.minimum(min_dists, dist[farthest], out=min_dists)
        
        upper_bound = np.max(min_dists)
        
        # Binary search for optimal radius
        left, right = 0, len(unique_dists) - 1
        best_centers = centers_idx.copy()
        
        while left <= right:
            mid = (left + right) // 2
            threshold = unique_dists[mid]
            
            if threshold >= upper_bound:
                right = mid - 1
                continue
            
            # Check if solution exists with this threshold using SAT
            sat = SATSolver('MiniCard')
            
            # Variables: node i is a center (1 to n)
            # Add constraint: at most k centers
            sat.add_atmost(list(range(1, n + 1)), k=k_centers)
            
            # For each node, at least one center within threshold distance
            for i in range(n):
                clause = []
                for j in range(n):
                    if dist[i, j] <= threshold:
                        clause.append(j + 1)  # SAT variables are 1-indexed
                if clause:
                    sat.add_clause(clause)
                else:
                    # No solution possible with this threshold
                    sat.delete()
                    break
            else:
                # Try to solve
                if sat.solve():
                    model = sat.get_model()
                    if model:
                        best_centers = [i for i in range(n) if (i + 1) in model]
                        upper_bound = threshold
                        right = mid - 1
                    else:
                        left = mid + 1
                else:
                    left = mid + 1
                sat.delete()
        
        # Convert indices back to node names
        return {nodes[idx] for idx in best_centers}