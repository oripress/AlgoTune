import networkx as nx
import numpy as np
import math
import random
import bisect
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        G_dict, k = problem
        
        # Build graph
        graph = nx.Graph()
        for v, adj in G_dict.items():
            for w, d in adj.items():
                graph.add_edge(v, w, weight=d)
        
        nodes = list(graph.nodes)
        n = len(nodes)
        if n == 0 or k == 0:
            return []
        if k >= n:
            return nodes
        
        # Handle disconnected graphs
        components = list(nx.connected_components(graph))
        if len(components) > k:
            return [next(iter(comp)) for comp in components[:k]]
        
        # Compute APSP using Dijkstra (faster for sparse graphs)
        dist_dict = {}
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        for node in nodes:
            dist_dict[node] = nx.single_source_dijkstra_path_length(graph, node, weight='weight')
        
        # Create distance matrix
        dist_matrix = np.full((n, n), np.inf)
        for i, u in enumerate(nodes):
            for v, d in dist_dict[u].items():
                j = node_to_idx[v]
                dist_matrix[i, j] = d
        np.fill_diagonal(dist_matrix, 0)
        
        # Get finite distances and sort
        finite_mask = np.isfinite(dist_matrix)
        if not np.any(finite_mask):
            return []
        distinct_d = np.unique(dist_matrix[finite_mask])
        distinct_d.sort()
        
        # Optimized greedy initialization
        min_dists = np.full(n, np.inf)
        centers_idx = []
        first_center = random.randint(0, n-1)
        centers_idx.append(first_center)
        min_dists = np.minimum(min_dists, dist_matrix[first_center])
        
        for _ in range(1, k):
            farthest = np.argmax(min_dists)
            centers_idx.append(farthest)
            min_dists = np.minimum(min_dists, dist_matrix[farthest])
        
        initial_obj = np.max(min_dists)
        best_solution = centers_idx
        
        # Binary search on distinct distances
        low_idx = 0
        high_idx = bisect.bisect_left(distinct_d, initial_obj)
        if high_idx >= len(distinct_d):
            high_idx = len(distinct_d) - 1
        
        while low_idx <= high_idx:
            mid_idx = (low_idx + high_idx) // 2
            mid_dist = distinct_d[mid_idx]
            
            # Precompute coverage mask for this distance
            mask = dist_matrix <= mid_dist
            
            # First try greedy coverage
            centers = self.greedy_cover(mask, k, n)
            if centers is not None:
                best_solution = centers
                high_idx = mid_idx - 1
            else:
                # If greedy fails, use SAT solver
                centers = self.sat_cover(mask, k, n)
                if centers is not None:
                    best_solution = centers
                    high_idx = mid_idx - 1
                else:
                    low_idx = mid_idx + 1
        
        return [nodes[i] for i in best_solution]
    
    def greedy_cover(self, mask, k, n):
        covered = np.zeros(n, dtype=bool)
        centers = []
        
        for _ in range(k):
            # Count uncovered nodes for each candidate center
            uncovered_counts = np.sum(mask & ~covered, axis=1)
            best_center = np.argmax(uncovered_counts)
            
            # Early termination if no coverage improvement
            if uncovered_counts[best_center] == 0:
                if np.all(covered):
                    return centers
                return None
                
            centers.append(best_center)
            covered = covered | mask[best_center]
            
            # Early termination if all covered
            if np.all(covered):
                return centers
                
        return centers if np.all(covered) else None
    
    def sat_cover(self, mask, k, n):
        # Build coverage sets from mask
        coverage_sets = [np.nonzero(mask[:, j])[0].tolist() for j in range(n)]
        
        # Create CP-SAT model
        model = cp_model.CpModel()
        x = [model.NewBoolVar(f'x_{i}') for i in range(n)]
        model.Add(sum(x) == k)  # Exactly k centers
        
        # Add coverage constraints
        for j in range(n):
            if not coverage_sets[j]:
                return None
            cover_vars = [x[i] for i in coverage_sets[j]]
            model.AddBoolOr(cover_vars)
        
        # Solve with time limit
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 0.5
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            return [i for i in range(n) if solver.Value(x[i])]
        return None