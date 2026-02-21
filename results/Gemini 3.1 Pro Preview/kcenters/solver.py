import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from ortools.sat.python import cp_model
from typing import Any
import bisect

class Solver:
    def solve(self, problem: tuple[dict[str, dict[str, float]], int], **kwargs) -> Any:
        G_dict, k = problem
        
        if not G_dict:
            return []
        if k == 0:
            return []
            
        nodes = list(G_dict.keys())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        if k >= n:
            return nodes
        # Build adjacency matrix
        row = []
        col = []
        data = []
        for u, neighbors in G_dict.items():
            u_idx = node_to_idx[u]
            for v, weight in neighbors.items():
                v_idx = node_to_idx[v]
                row.append(u_idx)
                col.append(v_idx)
                data.append(weight)
                
        adj_matrix = csr_matrix((data, (row, col)), shape=(n, n))
        dist_matrix = shortest_path(csgraph=adj_matrix, directed=False, return_predecessors=False)
        # Heuristic to find a good upper bound
        max_dists = np.max(dist_matrix, axis=1)
        first_center = int(np.argmin(max_dists))
        
        centers_idx = [first_center]
        min_dist_to_centers = dist_matrix[first_center].copy()
        
        for _ in range(1, k):
            next_center = int(np.argmax(min_dist_to_centers))
            centers_idx.append(next_center)
            min_dist_to_centers = np.minimum(min_dist_to_centers, dist_matrix[next_center])
            
        upper_bound = np.max(min_dist_to_centers)
        best_solution = [nodes[i] for i in centers_idx]
        
        # Get unique distances
        unique_dists = np.unique(dist_matrix)
        unique_dists = unique_dists[unique_dists != np.inf]
        
        # Filter distances based on upper bound and lower bound (U/2)
        lower_bound = upper_bound / 2.0
        idx_left = bisect.bisect_left(unique_dists, lower_bound)
        idx_right = bisect.bisect_right(unique_dists, upper_bound)
        unique_dists = unique_dists[idx_left:idx_right]
        
        # Binary search
        left = 0
        right = len(unique_dists) - 2 # -1 is the upper bound, which we know is feasible
        if right < 0:
            return best_solution
            
        while left <= right:
            mid = (left + right) // 2
            D = unique_dists[mid]
            
            covered_matrix = dist_matrix <= D
            
            # Fast greedy set cover heuristic
            uncovered = np.ones(n, dtype=bool)
            greedy_centers = []
            for _ in range(k):
                if not uncovered.any():
                    break
                cover_counts = np.logical_and(covered_matrix, uncovered).sum(axis=1)
                best_center = int(np.argmax(cover_counts))
                if cover_counts[best_center] == 0:
                    break
                greedy_centers.append(best_center)
                uncovered[covered_matrix[best_center]] = False
                
            if not uncovered.any():
                # Greedy heuristic found a valid cover
                best_solution = [nodes[i] for i in greedy_centers]
                right = mid - 1
                continue
            
            # Feasibility check: Is there a dominating set of size <= k for distance D?
            model = cp_model.CpModel()
            
            # Variables: x[i] = 1 if node i is a center
            x = [model.NewBoolVar('') for _ in range(n)]
            
            # Constraint: At most k centers
            model.Add(cp_model.LinearExpr.Sum(x) <= k)
            
            # Constraint: Every node must be covered
            for i in range(n):
                covered_by = [x[j] for j in np.flatnonzero(covered_matrix[i])]
                model.AddBoolOr(covered_by)
                
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
                
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            status = solver.Solve(model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                # Feasible, try smaller distance
                best_solution = [nodes[i] for i in range(n) if solver.Value(x[i])]
                right = mid - 1
            else:
                # Infeasible, try larger distance
                left = mid + 1
                
        return best_solution