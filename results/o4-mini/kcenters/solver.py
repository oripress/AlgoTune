import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import bisect
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs):
        G_dict, k = problem
        nodes = list(G_dict.keys())
        n = len(nodes)
        # trivial cases
        if n == 0 or k <= 0:
            return set()
        if k >= n:
            return set(nodes)
        # map nodes to indices and build adjacency lists
        idx = {node: i for i, node in enumerate(nodes)}
        row, col, data = [], [], []
        for u, nbrs in G_dict.items():
            ui = idx[u]
            for v, d in nbrs.items():
                vi = idx[v]
                row.append(ui); col.append(vi); data.append(d)
        # all-pairs shortest paths via SciPy Dijkstra
        mat = csr_matrix((data, (row, col)), shape=(n, n))
        dist = dijkstra(mat, directed=False, return_predecessors=False,
                        indices=range(n))
        # extract sorted unique finite distances
        flat = dist[np.isfinite(dist)]
        if flat.size:
            dists = np.unique(flat)
        else:
            dists = np.array([0.0], dtype=float)
        # greedy farthest-first heuristic to bound upper search radius
        centers = [0]
        dist_to_cent = dist[0].copy()
        for _ in range(1, k):
            nxt = int(np.argmax(dist_to_cent))
            centers.append(nxt)
            dist_to_cent = np.minimum(dist_to_cent, dist[nxt])
        radius_heur = float(dist_to_cent.max())
        hi = bisect.bisect_right(dists, radius_heur) - 1
        if hi < 0:
            hi = 0
        lo = 0
        best_centers = None
        # binary search on possible radii
        while lo <= hi:
            mid = (lo + hi) // 2
            R = dists[mid]
            # build CP-SAT decision model
            model = cp_model.CpModel()
            x = [model.NewBoolVar(f'x{i}') for i in range(n)]
            model.Add(sum(x) <= k)
            feasible = True
            for j in range(n):
                # require each node j to be covered by at least one center within R
                cover = [x[i] for i in range(n) if dist[i, j] <= R]
                if not cover:
                    feasible = False
                    break
                model.Add(sum(cover) >= 1)
            if feasible:
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = 0.5
                solver.parameters.num_search_workers = 8
                status = solver.Solve(model)
                if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                    best_centers = [i for i in range(n) if solver.Value(x[i]) == 1]
                    hi = mid - 1
                    continue
            lo = mid + 1
        # decode centers
        if best_centers is None:
            return set()
        return {nodes[i] for i in best_centers}