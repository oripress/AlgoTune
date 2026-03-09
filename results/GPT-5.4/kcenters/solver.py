from __future__ import annotations

import bisect
import heapq
import math
from typing import Any

try:
    from pysat.solvers import Solver as SATSolver
except Exception:  # pragma: no cover
    SATSolver = None

class Solver:
    def __init__(self) -> None:
        self._graph_id = None
        self._nodes: list[str] | None = None
        self._dist: list[list[float]] | None = None

    def solve(self, problem, **kwargs) -> Any:
        graph, k = problem
        nodes = list(graph)
        n = len(nodes)

        if n == 0 or k <= 0:
            return []
        if k >= n:
            return nodes[:]

        dist, nodes = self._prepare(graph)
        n = len(nodes)

        if k == 1:
            best = min(range(n), key=lambda i: max(dist[i]))
            return [nodes[best]]

        radii = self._candidate_radii(dist)
        heuristic = self._greedy_heuristic(dist, k)
        heuristic_obj = self._objective(dist, heuristic)
        cache: dict[float, tuple[bool, list[int] | None]] = {}

        hi = bisect.bisect_left(radii, heuristic_obj)
        if hi >= len(radii):
            hi = len(radii) - 1
        ok, _ = self._feasible(dist, radii[hi], k, cache, need_solution=False)
        if not ok:
            hi = len(radii) - 1

        lo = 0
        while lo < hi:
            mid = (lo + hi) // 2
            ok, _ = self._feasible(dist, radii[mid], k, cache, need_solution=False)
            if ok:
                hi = mid
            else:
                lo = mid + 1

        ok, solution = self._feasible(dist, radii[lo], k, cache, need_solution=True)
        if ok and solution is not None:
            return [nodes[i] for i in solution]
        return [nodes[i] for i in heuristic]

    def _prepare(self, graph: dict[str, dict[str, float]]) -> tuple[list[list[float]], list[str]]:
        graph_id = id(graph)
        if graph_id == self._graph_id and self._nodes is not None and self._dist is not None:
            return self._dist, self._nodes

        nodes = list(graph)
        idx = {u: i for i, u in enumerate(nodes)}
        n = len(nodes)

        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for u, nbrs in graph.items():
            ui = idx[u]
            row = adj[ui]
            for v, w in nbrs.items():
                vi = idx[v]
                fw = float(w)
                row.append((vi, fw))
                rows.append(ui)
                cols.append(vi)
                data.append(fw)

        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra

            mat = csr_matrix((data, (rows, cols)), shape=(n, n))
            dist = dijkstra(mat, directed=False, unweighted=False).tolist()
        except Exception:
            dist = self._apsp_sparse(adj)

        self._graph_id = graph_id
        self._nodes = nodes
        self._dist = dist
        return dist, nodes

    def _apsp_sparse(self, adj: list[list[tuple[int, float]]]) -> list[list[float]]:
        n = len(adj)
        inf = math.inf
        out = [[inf] * n for _ in range(n)]
        for s in range(n):
            ds = out[s]
            ds[s] = 0.0
            pq: list[tuple[float, int]] = [(0.0, s)]
            while pq:
                d, u = heapq.heappop(pq)
                if d != ds[u]:
                    continue
                for v, w in adj[u]:
                    nd = d + w
                    if nd < ds[v]:
                        ds[v] = nd
                        heapq.heappush(pq, (nd, v))
        return out

    def _apsp_dense(
        self,
        graph: dict[str, dict[str, float]],
        nodes: list[str],
        idx: dict[str, int],
    ) -> list[list[float]]:
        import numpy as np

        n = len(nodes)
        mat = np.full((n, n), np.inf, dtype=float)
        np.fill_diagonal(mat, 0.0)

        for u, nbrs in graph.items():
            ui = idx[u]
            row = mat[ui]
            for v, w in nbrs.items():
                vi = idx[v]
                fw = float(w)
                if fw < row[vi]:
                    row[vi] = fw

        for k in range(n):
            np.minimum(mat, mat[:, [k]] + mat[[k], :], out=mat)

        return mat.tolist()

    def _candidate_radii(self, dist: list[list[float]]) -> list[float]:
        vals = {0.0}
        inf = math.inf
        n = len(dist)
        for i in range(n):
            row = dist[i]
            for j in range(i + 1, n):
                d = row[j]
                if d < inf:
                    vals.add(d)
        return sorted(vals)

    def _greedy_heuristic(self, dist: list[list[float]], k: int) -> list[int]:
        n = len(dist)
        first = min(range(n), key=lambda i: max(dist[i]))
        centers = [first]
        chosen = [False] * n
        chosen[first] = True
        best = dist[first][:]

        for _ in range(1, k):
            nxt = -1
            farthest = -1.0
            for v in range(n):
                if not chosen[v] and best[v] > farthest:
                    farthest = best[v]
                    nxt = v
            if nxt < 0:
                break
            centers.append(nxt)
            chosen[nxt] = True
            row = dist[nxt]
            for v in range(n):
                if row[v] < best[v]:
                    best[v] = row[v]
        return centers

    def _objective(self, dist: list[list[float]], centers: list[int]) -> float:
        n = len(dist)
        best = [math.inf] * n
        for c in centers:
            row = dist[c]
            for v in range(n):
                dv = row[v]
                if dv < best[v]:
                    best[v] = dv
        return max(best)

    def _feasible(
        self,
        dist: list[list[float]],
        radius: float,
        k: int,
        cache: dict[float, tuple[bool, list[int] | None]],
        need_solution: bool,
    ) -> tuple[bool, list[int] | None]:
        hit = cache.get(radius)
        if hit is not None and (not need_solution or hit[1] is not None):
            return hit

        cover_masks = self._build_cover_masks(dist, radius)
        n = len(cover_masks)
        full_mask = (1 << n) - 1

        if self._component_lower_bound(cover_masks, n) > k:
            cache[radius] = (False, None)
            return cache[radius]

        masks, original = self._reduce_masks(cover_masks)

        if len(masks) <= k:
            cache[radius] = (True, original[:])
            return cache[radius]

        for i, mask in enumerate(masks):
            if mask == full_mask:
                cache[radius] = (True, [original[i]])
                return cache[radius]

        solution: list[int] | None = None
        if len(masks) <= 60 and k <= 8:
            local = self._branch_search(masks, k)
            if local is not None:
                solution = [original[i] for i in local]

        if solution is None:
            solution = self._solve_sat(masks, original, n, k)

        cache[radius] = (solution is not None, solution)
        return cache[radius]

    def _build_cover_masks(self, dist: list[list[float]], radius: float) -> list[int]:
        n = len(dist)
        masks = [0] * n
        for i in range(n):
            row = dist[i]
            mask = 0
            for j in range(n):
                if row[j] <= radius:
                    mask |= 1 << j
            masks[i] = mask
        return masks

    def _reduce_masks(self, masks: list[int]) -> tuple[list[int], list[int]]:
        first_idx: dict[int, int] = {}
        for i, mask in enumerate(masks):
            first_idx.setdefault(mask, i)

        items = [(mask.bit_count(), mask, i) for mask, i in first_idx.items()]
        items.sort(reverse=True)

        kept_masks: list[int] = []
        kept_idx: list[int] = []
        for _, mask, i in items:
            dominated = False
            for prev in kept_masks:
                if (mask | prev) == prev:
                    dominated = True
                    break
            if not dominated:
                kept_masks.append(mask)
                kept_idx.append(i)
        return kept_masks, kept_idx

    def _component_lower_bound(self, masks: list[int], n: int) -> int:
        unseen = (1 << n) - 1
        components = 0
        while unseen:
            components += 1
            seed = unseen & -unseen
            unseen ^= seed
            stack = seed
            while stack:
                bit = stack & -stack
                stack ^= bit
                u = bit.bit_length() - 1
                add = masks[u] & unseen
                if add:
                    unseen &= ~add
                    stack |= add
        return components

    def _branch_search(self, masks: list[int], k: int) -> list[int] | None:
        n = 0
        for mask in masks:
            if mask:
                n = max(n, mask.bit_length())
        full_mask = (1 << n) - 1

        covered_by: list[list[int]] = [[] for _ in range(n)]
        gains = [mask.bit_count() for mask in masks]
        for ci, mask in enumerate(masks):
            x = mask
            while x:
                bit = x & -x
                v = bit.bit_length() - 1
                covered_by[v].append(ci)
                x ^= bit

        for lst in covered_by:
            if not lst:
                return None

        dead: set[tuple[int, int]] = set()

        def dfs(uncovered: int, left: int) -> tuple[int, ...] | None:
            if uncovered == 0:
                return ()
            if left == 0:
                return None

            state = (uncovered, left)
            if state in dead:
                return None

            remaining = uncovered.bit_count()
            max_gain = 0
            uu = uncovered
            seen_centers: set[int] = set()
            while uu:
                bit = uu & -uu
                v = bit.bit_length() - 1
                for c in covered_by[v]:
                    if c in seen_centers:
                        continue
                    seen_centers.add(c)
                    gain = (masks[c] & uncovered).bit_count()
                    if gain > max_gain:
                        max_gain = gain
                uu ^= bit
            if max_gain * left < remaining:
                dead.add(state)
                return None

            pivot = -1
            best_len = 10**9
            uu = uncovered
            while uu:
                bit = uu & -uu
                v = bit.bit_length() - 1
                cand_len = len(covered_by[v])
                if cand_len < best_len:
                    best_len = cand_len
                    pivot = v
                    if cand_len <= 1:
                        break
                uu ^= bit

            options = sorted(
                covered_by[pivot],
                key=lambda c: ((masks[c] & uncovered).bit_count(), gains[c]),
                reverse=True,
            )

            seen_next: set[int] = set()
            for c in options:
                new_uncovered = uncovered & ~masks[c]
                if new_uncovered == uncovered or new_uncovered in seen_next:
                    continue
                seen_next.add(new_uncovered)
                tail = dfs(new_uncovered, left - 1)
                if tail is not None:
                    return (c,) + tail

            dead.add(state)
            return None

        result = dfs(full_mask, k)
        return None if result is None else list(result)

    def _solve_sat(
        self,
        masks: list[int],
        original: list[int],
        n: int,
        k: int,
    ) -> list[int] | None:
        if SATSolver is None:
            return None

        clauses = [[] for _ in range(n)]
        for var, mask in enumerate(masks, start=1):
            x = mask
            while x:
                bit = x & -x
                v = bit.bit_length() - 1
                clauses[v].append(var)
                x ^= bit

        for clause in clauses:
            if not clause:
                return None

        solver = SATSolver(name="MiniCard")
        try:
            solver.add_atmost(list(range(1, len(masks) + 1)), k=k)
            for clause in clauses:
                solver.add_clause(clause)
            if not solver.solve():
                return None
            model = solver.get_model() or []
        finally:
            try:
                solver.delete()
            except Exception:
                pass

        selected = {v for v in model if v > 0}
        return [original[var - 1] for var in range(1, len(masks) + 1) if var in selected]