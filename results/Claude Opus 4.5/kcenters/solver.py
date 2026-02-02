import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from ortools.sat.python import cp_model
from itertools import combinations
from math import comb

class Solver:
    def solve(self, problem, **kwargs):
        G_dict, k = problem
        
        if not G_dict:
            return []
        
        nodes = list(G_dict.keys())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        if k == 0:
            return []
        
        if k >= n:
            return nodes
        
        # Build sparse adjacency matrix
        row, col, data = [], [], []
        for u, adj in G_dict.items():
            i = node_to_idx[u]
            for v, d in adj.items():
                j = node_to_idx[v]
                row.append(i)
                col.append(j)
                data.append(d)
        
        adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))
        dist_matrix = dijkstra(adj_matrix, directed=False)
        
        def compute_obj(centers):
            return float(np.max(np.min(dist_matrix[centers], axis=0)))
        
        # Greedy heuristic (vectorized)
        def greedy_kcenter():
            centers = []
            min_dist = np.full(n, np.inf)
            ecc = np.nanmax(np.where(np.isinf(dist_matrix), np.nan, dist_matrix), axis=1)
            first = int(np.nanargmin(ecc))
            centers.append(first)
            min_dist = dist_matrix[first].copy()
            
            for _ in range(k - 1):
                farthest = int(np.argmax(min_dist))
                centers.append(farthest)
                np.minimum(min_dist, dist_matrix[farthest], out=min_dist)
            
            return centers, float(np.max(min_dist))
        
        best_centers, best_obj = greedy_kcenter()
        
        # For small problems, try enumeration
        try:
            num_combos = comb(n, k)
        except:
            num_combos = float('inf')
        
        if num_combos <= 100000:
            for combo in combinations(range(n), k):
                obj = compute_obj(list(combo))
                if obj < best_obj:
                    best_obj = obj
                    best_centers = list(combo)
            return [nodes[i] for i in best_centers]
        
        # Get candidate distances
        all_dist_flat = dist_matrix.flatten()
        unique_dists = np.sort(np.unique(all_dist_flat[np.isfinite(all_dist_flat)]))
        
        def can_cover_with_k(limit):
            """Check if k centers can cover all nodes within distance limit using CP-SAT"""
            model = cp_model.CpModel()
            x = [model.NewBoolVar(f'x_{i}') for i in range(n)]
            
            # At most k centers
            model.Add(sum(x) <= k)
            
            # Each node must be covered
            for v in range(n):
                covering = [x[u] for u in range(n) if dist_matrix[u, v] <= limit]
                if not covering:
                    return False, None
                model.AddBoolOr(covering)
            
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 0.5
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                centers = [i for i in range(n) if solver.Value(x[i]) == 1]
                return True, centers
            return False, None
        
        # Binary search on distances
        left, right = 0, len(unique_dists) - 1
        result_idx = right
        
        # Find the smallest index where we can cover with k centers
        while left <= right:
            mid = (left + right) // 2
            feasible, centers = can_cover_with_k(unique_dists[mid])
            if feasible:
                if unique_dists[mid] < best_obj:
                    best_obj = unique_dists[mid]
                    obj = compute_obj(centers)
                    if obj <= best_obj:
                        best_centers = centers
                result_idx = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return [nodes[i] for i in best_centers]