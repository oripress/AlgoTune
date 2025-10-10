import bisect
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from pysat.solvers import Solver as SATSolver

class Distances:
    """Helper class to manage distances in the graph using scipy."""
    
    def __init__(self, G_dict):
        # Create node to index mapping
        self._vertices = list(G_dict.keys())
        self._node_to_idx = {node: i for i, node in enumerate(self._vertices)}
        n = len(self._vertices)
        
        # Build adjacency matrix
        data, row, col = [], [], []
        for u, adj in G_dict.items():
            u_idx = self._node_to_idx[u]
            for v, weight in adj.items():
                v_idx = self._node_to_idx[v]
                data.append(weight)
                row.append(u_idx)
                col.append(v_idx)
        
        # Create sparse matrix and compute all-pairs shortest paths
        adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))
        self._dist_matrix = floyd_warshall(adj_matrix, directed=False)
        self._unique_distances = None
    
    def all_vertices(self):
        return self._vertices
    
    def dist(self, u, v):
        u_idx = self._node_to_idx[u]
        v_idx = self._node_to_idx[v]
        return self._dist_matrix[u_idx, v_idx]
    
    def vertices_in_range(self, v, limit):
        v_idx = self._node_to_idx[v]
        indices = np.where(self._dist_matrix[v_idx] <= limit)[0]
        return [self._vertices[i] for i in indices]
    
    def max_dist(self, centers):
        """Return the maximum distance from any vertex to its nearest center."""
        center_indices = [self._node_to_idx[c] for c in centers]
        # For each vertex, find minimum distance to any center
        min_dists = np.min(self._dist_matrix[:, center_indices], axis=1)
        return np.max(min_dists)
    
    def sorted_distances(self):
        """Return sorted list of all unique distances in the graph."""
        if self._unique_distances is None:
            # Get all unique non-zero distances
            unique = np.unique(self._dist_matrix[self._dist_matrix > 0])
            self._unique_distances = unique.tolist()
        return self._unique_distances

class KCenterDecisionVariant:
    """SAT-based decision variant for k-centers."""
    
    def __init__(self, distances, k):
        self.distances = distances
        self._node_vars = {
            node: i for i, node in enumerate(self.distances.all_vertices(), start=1)
        }
        self._sat_solver = SATSolver("MiniCard")
        self._sat_solver.add_atmost(list(self._node_vars.values()), k=k)
        self._solution = None
    
    def limit_distance(self, limit):
        for v in self.distances.all_vertices():
            clause = [
                self._node_vars[u] for u in self.distances.vertices_in_range(v, limit)
            ]
            self._sat_solver.add_clause(clause)
    
    def solve(self):
        if not self._sat_solver.solve():
            return None
        model = self._sat_solver.get_model()
        if model is None:
            return None
        self._solution = [node for node, var in self._node_vars.items() if var in model]
        return self._solution
    
    def get_solution(self):
        if self._solution is None:
            raise ValueError("No solution available.")
        return self._solution

class Solver:
    def solve(self, problem, **kwargs):
        """
        Solves the k-centers problem for the given graph instance.
        
        Args:
            problem (tuple): A tuple (G, k) where G is the weighted graph dictionary and k is the number of centers.
        
        Returns:
            set: Set of node IDs chosen as centers.
        """
        G_dict, k = problem
        
        # Handle edge cases
        n = len(G_dict)
        if k == 0:
            return set()
        if k >= n:
            return set(G_dict.keys())
        
        distances = Distances(G_dict)
        centers = self._solve_heur(distances, k)
        
        # SAT-based refinement
        obj = distances.max_dist(centers)
        decision_variant = KCenterDecisionVariant(distances, k)
        sorted_dists = distances.sorted_distances()
        index = bisect.bisect_left(sorted_dists, obj)
        sorted_dists = sorted_dists[:index]
        
        if sorted_dists:
            decision_variant.limit_distance(sorted_dists[-1])
            while decision_variant.solve() is not None:
                centers = decision_variant.get_solution()
                obj = distances.max_dist(centers)
                index = bisect.bisect_left(sorted_dists, obj)
                sorted_dists = sorted_dists[:index]
                if not sorted_dists:
                    break
                decision_variant.limit_distance(sorted_dists.pop())
        
        return set(centers)
    
    def _solve_heur(self, distances, k):
        """Greedy heuristic for k-centers."""
        vertices = distances.all_vertices()
        n = len(vertices)
        remaining = set(range(n))
        
        if not remaining or k == 0:
            return []
        
        # Pick first center - node that minimizes maximum distance to all others
        first_idx = min(
            remaining,
            key=lambda c: max(distances._dist_matrix[c, u] for u in remaining),
        )
        remaining.remove(first_idx)
        center_indices = [first_idx]
        
        # Greedily add remaining centers
        while len(center_indices) < k and remaining:
            # Find node that is farthest from all current centers
            max_dist_idx = max(
                remaining,
                key=lambda v: min(distances._dist_matrix[v, c] for c in center_indices),
            )
            remaining.remove(max_dist_idx)
            center_indices.append(max_dist_idx)
        
        return [vertices[i] for i in center_indices]