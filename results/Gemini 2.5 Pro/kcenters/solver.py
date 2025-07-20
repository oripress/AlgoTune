import numpy as np
from numba import njit
from pysat.solvers import Solver as SATSolver
from typing import Any, Dict, List, Set, Tuple

class Solver:
    """
    Solves the k-centers problem by first computing all-pairs shortest paths
    using a Numba-accelerated Floyd-Warshall algorithm. It then uses a binary
    search over the possible distances to find the optimal radius. The decision
    problem for a given radius is solved using a SAT solver. A greedy heuristic
    is used to establish a good initial upper bound for the binary search.
    """

    def solve(self, problem: Tuple[Dict[str, Dict[str, float]], int], **kwargs: Any) -> Set[str]:
        """
        Finds an optimal set of k centers.

        Args:
            problem: A tuple containing the graph dictionary and the number of centers k.

        Returns:
            A set of node names representing the chosen centers.
        """
        G_dict, k = problem

        if k <= 0:
            return set()
        
        nodes = list(G_dict.keys())
        if not nodes:
            return set()
        
        if k >= len(nodes):
            return set(nodes)

        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        n = len(nodes)

        adj_matrix = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(adj_matrix, 0)

        for u, neighbors in G_dict.items():
            u_idx = node_to_idx[u]
            for v, weight in neighbors.items():
                if v in node_to_idx:
                    v_idx = node_to_idx[v]
                    adj_matrix[u_idx, v_idx] = float(weight)
        
        dist_matrix = self._floyd_warshall(adj_matrix)

        unique_distances = np.unique(dist_matrix[np.isfinite(dist_matrix)])
        
        # Find a good initial solution using a fast heuristic
        heur_centers_indices = self._solve_heur(k, n, dist_matrix)
        
        # Calculate the objective value (max radius) for the heuristic solution
        max_dist_heur = 0.0
        if heur_centers_indices:
            min_dists_to_centers = np.min(dist_matrix[:, heur_centers_indices], axis=1)
            max_dist_heur = np.max(min_dists_to_centers)

        # Binary search for the optimal radius
        low = 0
        # Use the heuristic result as the upper bound for the search
        high = np.searchsorted(unique_distances, max_dist_heur, side='right') -1
        if high < 0: high = 0
        
        final_centers_indices = heur_centers_indices

        while low <= high:
            mid = low + (high - low) // 2
            if mid >= len(unique_distances):
                break
            radius = unique_distances[mid]
            
            current_centers = self._check(n, k, dist_matrix, radius)
            
            if current_centers is not None:
                # Found a valid solution, try for a smaller radius
                final_centers_indices = current_centers
                if mid == 0:
                    break
                high = mid - 1
            else:
                # No solution for this radius, need a larger one
                low = mid + 1
        
        return {idx_to_node[i] for i in final_centers_indices}

    def _solve_heur(self, k: int, n: int, dist_matrix: np.ndarray) -> List[int]:
        """A fast 2-approximation greedy heuristic."""
        # Start with the node that is the 1-center of the graph
        first_center = np.argmin(np.max(dist_matrix, axis=1))
        
        centers = [first_center]
        min_dists = dist_matrix[first_center, :].copy()
        
        while len(centers) < k:
            farthest_node = np.argmax(min_dists)
            centers.append(farthest_node)
            min_dists = np.minimum(min_dists, dist_matrix[farthest_node, :])
            
        return centers

    def _check(self, n: int, k: int, dist_matrix: np.ndarray, radius: float) -> List[int] | None:
        """Checks if a k-cover exists for a given radius using a SAT solver."""
        with SATSolver(name='minicard') as solver:
            # Variables 1 to n represent the nodes
            node_vars = list(range(1, n + 1))
            # Cardinality constraint: at most k centers can be chosen
            solver.add_atmost(lits=node_vars, k=k)
            
            # Coverage constraint: each node must be covered by a center
            for i in range(n):
                # Find all potential centers for node i (nodes within `radius`)
                potential_centers = (np.where(dist_matrix[i, :] <= radius + 1e-9)[0] + 1).tolist()
                
                # If a node cannot be covered by any other node, no solution exists for this radius
                if not potential_centers:
                    return None
                
                # Add clause: one of the potential centers must be chosen
                solver.add_clause(potential_centers)
                
            if solver.solve():
                model = solver.get_model()
                if model is None: return None
                # Return the indices of the chosen centers
                return [var - 1 for var in model if var > 0]
            else:
                return None

    @staticmethod
    @njit(fastmath=True)
    def _floyd_warshall(dist: np.ndarray) -> np.ndarray:
        """Numba-jitted Floyd-Warshall algorithm for all-pairs shortest paths."""
        n = dist.shape[0]
        for k_ in range(n):
            for i in range(n):
                for j in range(n):
                    d = dist[i, k_] + dist[k_, j]
                    if d < dist[i, j]:
                        dist[i, j] = d
        return dist