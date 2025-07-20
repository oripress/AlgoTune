import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from bisect import bisect_left
from functools import lru_cache

class Solver:
    def solve(self, problem, **kwargs):
        G_dict, k_input = problem
        nodes = list(G_dict.keys())
        n = len(nodes)
        k = k_input
        if n == 0 or k <= 0:
            return []
        if k >= n:
            return nodes.copy()

        # build graph and APSP distances
        idx = {u: i for i, u in enumerate(nodes)}
        rows = []; cols = []; data = []
        for u, nbrs in G_dict.items():
            ui = idx[u]
            for v, w in nbrs.items():
                rows.append(ui); cols.append(idx[v]); data.append(w)
        A = csr_matrix((data, (rows, cols)), shape=(n, n))
        dist_mat = dijkstra(A, directed=False)
        dist = dist_mat.tolist()

        # 2-approx greedy heuristic to bound radius
        maxd = [max(row) for row in dist]
        first = min(range(n), key=lambda i: maxd[i])
        centers0 = [first]
        covered = dist[first].copy()
        for _ in range(1, min(k, n)):
            i = max(range(n), key=lambda i: covered[i])
            centers0.append(i)
            di = dist[i]
            for j in range(n):
                dj = di[j]
                if dj < covered[j]:
                    covered[j] = dj
        heur_obj = max(covered)

        # unique sorted distances
        flat = {d for row in dist for d in row}
        ds = sorted(flat)
        lo, hi = 0, bisect_left(ds, heur_obj)

        full_mask = (1 << n) - 1
        bits = [1 << j for j in range(n)]
        range_n = range(n)
        dist_local = dist

        coverage = [0] * n
        coverers = [[] for _ in range(n)]
        best_idx = None

        # binary search on radius
        while lo <= hi:
            mid = (lo + hi) >> 1
            R = ds[mid]
            for j in range_n:
                coverers[j].clear()
            for i in range_n:
                mask = 0
                di = dist_local[i]
                for j in range_n:
                    if di[j] <= R:
                        mask |= bits[j]
                        coverers[j].append(i)
                coverage[i] = mask
            # infeasible if any uncovered node
            feasible = True
            for j in range_n:
                if not coverers[j]:
                    feasible = False
                    break
            if not feasible:
                lo = mid + 1
                continue
            @lru_cache(None)
            def dfs(uncovered, depth):
                if uncovered == 0:
                    return ()
                if depth == k:
                    return None
                j0 = (uncovered & -uncovered).bit_length() - 1
                for c in coverers[j0]:
                    sol = dfs(uncovered & ~coverage[c], depth + 1)
                    if sol is not None:
                        return (c,) + sol
                return None

            sol = dfs(full_mask, 0)
            if sol is not None:
                best_idx = sol
                hi = mid - 1
            else:
                lo = mid + 1

        if best_idx is None:
            best_idx = tuple(centers0)
        return [nodes[i] for i in best_idx]