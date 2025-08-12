import numpy as np
import networkx as nx
from ortools.sat.python import cp_model
from collections import defaultdict
import heapq
import numba

@numba.jit(nopython=True)
def floyd_warshall_numba(dist_matrix, n):
    """Numba-optimized Floyd-Warshall algorithm."""
    for k in range(n):
        for i in range(n):
            if dist_matrix[i, k] == 1e20:  # Using 1e20 instead of inf for numba
                continue
            for j in range(n):
                if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                    dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
    return dist_matrix

class Solver:
    def solve(self, problem, **kwargs):
        """Ultra-optimized solver for Vertex K-Center problem."""
        G_dict, k = problem
        
        # Handle edge cases immediately
        if not G_dict:
            return []
        if k == 0:
            return []
        if k >= len(G_dict):
            return list(G_dict.keys())
        
        # Get vertices
        vertices = list(G_dict.keys())
        n = len(vertices)
        vertex_to_idx = {v: i for i, v in enumerate(vertices)}
        
        # Precompute all-pairs shortest paths
        if n <= 150:
            # Use numba-optimized Floyd-Warshall for small graphs
            dist_matrix = np.full((n, n), 1e20, dtype=np.float64)
            
            # Initialize diagonal and direct edges
            np.fill_diagonal(dist_matrix, 0)
            
            for v, adj in G_dict.items():
                i = vertex_to_idx[v]
                for w, d in adj.items():
                    j = vertex_to_idx[w]
                    if d < dist_matrix[i, j]:
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d
            
            # Run numba-optimized Floyd-Warshall
            dist_matrix = floyd_warshall_numba(dist_matrix, n)
            
            # Convert to dictionary format
            dist_dict = {}
            for i, v in enumerate(vertices):
                dist_dict[v] = {}
                for j, w in enumerate(vertices):
                    dist_dict[v][w] = dist_matrix[i, j]
        else:
            # Use NetworkX for larger graphs
            graph = nx.Graph()
            for v, adj in G_dict.items():
                for w, d in adj.items():
                    graph.add_edge(v, w, weight=d)
            dist_dict = dict(nx.all_pairs_dijkstra_path_length(graph))
        
        # Ultra-fast Gonzalez heuristic
        # Precompute eccentricity using numpy for small graphs
        if n <= 150:
            eccentricity = np.max(dist_matrix, axis=1)
            first_center_idx = np.argmin(eccentricity)
            first_center = vertices[first_center_idx]
        else:
            eccentricity = {}
            for v in vertices:
                eccentricity[v] = max(dist_dict[v][u] for u in vertices)
            first_center = min(vertices, key=lambda v: eccentricity[v])
        
        centers = [first_center]
        min_dist = np.full(n, 1e20) if n <= 150 else {v: float('inf') for v in vertices}
        
        # Update distances from first center
        if n <= 150:
            for j, w in enumerate(vertices):
                min_dist[j] = dist_matrix[first_center_idx, j]
        else:
            for v in vertices:
                min_dist[v] = dist_dict[first_center][v]
        
        # Fast center selection
        for _ in range(1, k):
            if n <= 150:
                # Use numpy for small graphs
                next_center_idx = np.argmax(min_dist)
                next_center = vertices[next_center_idx]
                centers.append(next_center)
                
                # Update distances
                for j in range(n):
                    d = dist_matrix[next_center_idx, j]
                    if d < min_dist[j]:
                        min_dist[j] = d
            else:
                # Use dictionary for large graphs
                next_center = max(vertices, key=lambda v: min_dist[v])
                centers.append(next_center)
                
                # Update distances
                for v in vertices:
                    d = dist_dict[next_center][v]
                    if d < min_dist[v]:
                        min_dist[v] = d
        
        # Get initial radius
        if n <= 150:
            initial_radius = np.max(min_dist)
        else:
            initial_radius = max(min_dist.values())
        
        # Collect distinct distances efficiently
        if n <= 150:
            # Use numpy for small graphs
            unique_distances = np.unique(dist_matrix[dist_matrix <= initial_radius])
            distinct_distances = sorted(unique_distances)
        else:
            # Use set for large graphs
            distance_set = set()
            for u in vertices:
                for v in vertices:
                    d = dist_dict[u][v]
                    if d <= initial_radius:
                        distance_set.add(d)
            distinct_distances = sorted(distance_set)
        
        if not distinct_distances:
            distinct_distances = [initial_radius]
        
        # Ultra-fast decision function
        def decision(r):
            model = cp_model.CpModel()
            x = {v: model.NewBoolVar(f'x_{v}') for v in vertices}
            
            # Constraint: exactly k centers
            model.Add(sum(x[v] for v in vertices) == k)
            
            # Precompute coverage matrix for small graphs
            if n <= 150:
                coverage_matrix = dist_matrix <= r
                for i in range(n):
                    if not np.any(coverage_matrix[i]):
                        return False, None
                    model.Add(sum(x[vertices[j]] for j in range(n) if coverage_matrix[i, j]) >= 1)
            else:
                # Use dictionary for large graphs
                for u in vertices:
                    covering_centers = []
                    for v in vertices:
                        if dist_dict[u][v] <= r:
                            covering_centers.append(v)
                    if not covering_centers:
                        return False, None
                    model.Add(sum(x[v] for v in covering_centers) >= 1)
            
            # Fast solver configuration
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 0.5  # Very aggressive timeout
            solver.parameters.num_search_workers = 1
            solver.parameters.log_search_progress = False
            
            status = solver.Solve(model)
            
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                solution_centers = [v for v in vertices if solver.Value(x[v])]
                return True, solution_centers
            return False, None
        
        # Binary search with early termination
        low, high = 0, len(distinct_distances) - 1
        best_centers = centers
        
        while low <= high:
            mid = (low + high) // 2
            r = distinct_distances[mid]
            feasible, sol_centers = decision(r)
            
            if feasible:
                best_centers = sol_centers
                high = mid - 1
            else:
                low = mid + 1
        
        return best_centers