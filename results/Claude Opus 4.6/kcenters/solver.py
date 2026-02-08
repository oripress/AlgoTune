import numpy as np
import bisect
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from pysat.solvers import Solver as SATSolver


class Solver:
    def solve(self, problem, **kwargs):
        G_dict, k = problem
        nodes = list(G_dict.keys())
        n = len(nodes)
        
        if n == 0 or k == 0:
            return []
        if k >= n:
            return nodes
        
        node_idx = {nd: i for i, nd in enumerate(nodes)}
        
        # Build sparse adjacency matrix for scipy
        rows, cols, wts = [], [], []
        for u in G_dict:
            ui = node_idx[u]
            for v, w in G_dict[u].items():
                vi = node_idx[v]
                rows.append(ui)
                cols.append(vi)
                wts.append(float(w))
        
        mat = csr_matrix((wts, (rows, cols)), shape=(n, n))
        dist_matrix = shortest_path(mat, directed=False)
        
        # Precompute sorted neighbors by distance for each node
        sorted_neighbors = []
        for i in range(n):
            neighbors = []
            for j in range(n):
                d = dist_matrix[i, j]
                if np.isfinite(d):
                    neighbors.append((d, j))
            neighbors.sort()
            sorted_neighbors.append(neighbors)
        
        # Get all unique distances sorted
        all_dists = set()
        for i in range(n):
            for d, j in sorted_neighbors[i]:
                if d > 0:
                    all_dists.add(d)
        sorted_dists = sorted(all_dists)
        
        if not sorted_dists:
            return nodes[:k]
        
        # Greedy farthest-first heuristic
        safe_dist = np.where(np.isinf(dist_matrix), 1e18, dist_matrix)
        first = int(np.argmin(safe_dist.max(axis=1)))
        centers = [first]
        min_to_center = safe_dist[first].copy()
        
        for _ in range(k - 1):
            farthest = int(np.argmax(min_to_center))
            if min_to_center[farthest] <= 0:
                break
            centers.append(farthest)
            np.minimum(min_to_center, safe_dist[farthest], out=min_to_center)
        
        # Compute objective
        heur_obj = float(np.max(min_to_center))
        best_centers = list(centers)
        
        # Incremental SAT approach (like reference)
        sat = SATSolver("MiniCard")
        vl = list(range(1, n + 1))
        sat.add_atmost(vl, k=k)
        
        idx = bisect.bisect_left(sorted_dists, heur_obj)
        candidates = sorted_dists[:idx]
        
        if not candidates:
            sat.delete()
            return [nodes[c] for c in best_centers]
        
        # Add constraint for largest candidate
        def add_constraint(limit):
            for i in range(n):
                clause = []
                for d, j in sorted_neighbors[i]:
                    if d <= limit + 1e-9:
                        clause.append(j + 1)
                    else:
                        break
                if clause:
                    sat.add_clause(clause)
        
        add_constraint(candidates[-1])
        
        while sat.solve():
            model = sat.get_model()
            new_centers = [x - 1 for x in model if 0 < x <= n]
            best_centers = new_centers
            
            # Compute objective
            obj = float(np.max(np.min(dist_matrix[:, best_centers], axis=1)))
            
            idx = bisect.bisect_left(candidates, obj)
            candidates = candidates[:idx]
            if not candidates:
                break
            add_constraint(candidates.pop())
        
        sat.delete()
        return [nodes[c] for c in best_centers]