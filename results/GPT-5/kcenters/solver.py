from __future__ import annotations

from typing import Any, List, Tuple
import math
import heapq
from bisect import bisect_left, bisect_right

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Exact k-center on weighted undirected graphs using APSP + incremental SAT feasibility.
        Input: (G: dict[str, dict[str, float]], k: int)
        Output: list[str] of selected centers (size <= k)
        """
        G_dict, k = problem  # type: ignore[assignment]
        if not isinstance(G_dict, dict):
            raise TypeError("Graph must be a dict of dicts")
        if not isinstance(k, int):
            raise TypeError("k must be an int")

        nodes = list(G_dict.keys())
        n = len(nodes)

        # Handle trivial cases
        if n == 0:
            if k == 0:
                return []
            raise ValueError(f"Cannot find {k} centers in an empty graph.")
        if k == 0:
            return []
        if k >= n:
            # Select all nodes (or any n nodes) to get radius 0
            return nodes

        # Map nodes to indices
        idx_of = {v: i for i, v in enumerate(nodes)}

        # Build adjacency list
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for u_name, nbrs in G_dict.items():
            u = idx_of[u_name]
            for v_name, w in nbrs.items():
                v = idx_of.get(v_name)
                if v is None:
                    continue
                adj[u].append((v, float(w)))

        # All-pairs shortest paths via repeated Dijkstra
        INF = math.inf
        dist: List[List[float]] = [[INF] * n for _ in range(n)]
        for s in range(n):
            row = dist[s]
            row[s] = 0.0
            pq: List[Tuple[float, int]] = [(0.0, s)]
            heappop = heapq.heappop
            heappush = heapq.heappush
            while pq:
                d, u = heappop(pq)
                if d != row[u]:
                    continue
                for v, w in adj[u]:
                    nd = d + w
                    if nd < row[v]:
                        row[v] = nd
                        heappush(pq, (nd, v))

        # Precompute per-vertex sorted coverage lists (center distances to this vertex)
        cover_dists: List[List[float]] = []
        cover_vars: List[List[int]] = []
        unique_dset = set()
        for v in range(n):
            arr = [(dist[u][v], u) for u in range(n) if dist[u][v] < INF]
            arr.sort(key=lambda x: x[0])
            dists_v = [d for d, _ in arr]
            vars_v = [u + 1 for _, u in arr]  # CNF variables are 1-based
            cover_dists.append(dists_v)
            cover_vars.append(vars_v)
            unique_dset.update(dists_v)

        unique_dists = sorted(unique_dset)

        # Greedy farthest-first heuristic to get an initial feasible solution and its radius
        ecc = [max(row) for row in dist]
        first_center = min(range(n), key=lambda i: ecc[i])
        centers_idx = [first_center]
        mindist = dist[first_center][:]  # copy
        while len(centers_idx) < k and len(centers_idx) < n:
            # pick farthest from current centers
            next_center = max(range(n), key=lambda j: mindist[j])
            if next_center in centers_idx:
                break
            centers_idx.append(next_center)
            row = dist[next_center]
            for j in range(n):
                dj = row[j]
                if dj < mindist[j]:
                    mindist[j] = dj

        def max_dist_of_centers(centers: List[int]) -> float:
            md = 0.0
            for v in range(n):
                best = INF
                for c in centers:
                    dc = dist[c][v]
                    if dc < best:
                        best = dc
                if best > md:
                    md = best
            return md

        heur_obj = max_dist_of_centers(centers_idx)

        # Candidate radii strictly less than the heuristic objective
        index_end = bisect_left(unique_dists, heur_obj)
        if index_end <= 0:
            # No smaller candidate; heuristic is optimal or only INF improvements possible
            return [nodes[i] for i in centers_idx]

        candidates = unique_dists[:index_end]  # strictly less than heuristic

        # Incremental SAT decision variant
        try:
            from pysat.solvers import Solver as SATSolver  # type: ignore
            sat_solver = SATSolver(name="MiniCard")
            use_atmost = True
            try:
                sat_solver.add_atmost(list(range(1, n + 1)), k=k)  # type: ignore[attr-defined]
            except Exception:
                use_atmost = False
        except Exception:
            from pysat.solvers import Solver as SATSolver  # type: ignore
            sat_solver = SATSolver(name="m22")
            use_atmost = False

        if not use_atmost:
            # Encode at-most-k using sequential counter
            try:
                from pysat.card import CardEnc, EncType  # type: ignore
                cnf = CardEnc.atmost(lits=list(range(1, n + 1)), bound=k, encoding=EncType.seqcounter)
                sat_solver.append_formula(cnf.clauses)
            except Exception:
                # Fallback: pairwise at-most-k (O(n^2), but n likely small/moderate)
                lits = list(range(1, n + 1))
                for i in range(len(lits)):
                    ai = -lits[i]
                    for j in range(i + 1, len(lits)):
                        sat_solver.add_clause([ai, -lits[j]])

        # Helper to add coverage constraints up to radius R
        EPS = 1e-12

        def add_limit(R: float) -> None:
            # For each vertex v, require at least one center within distance <= R
            for v in range(n):
                dlist = cover_dists[v]
                idx = bisect_right(dlist, R + EPS)
                if idx <= 0:
                    # No center covers v within R: add empty clause to force UNSAT early
                    sat_solver.add_clause([])
                else:
                    clause = cover_vars[v][:idx]
                    sat_solver.add_clause(clause)

        # Start from the largest candidate < heuristic and go down while SAT
        last_sat_solution: List[int] = centers_idx[:]  # ensure we have something valid to return
        last_sat_obj = heur_obj

        # Add initial limit
        add_limit(candidates[-1])

        # Solve iteratively, tightening the radius
        # We pop decreasing candidates; on each SAT, compute true objective and cut candidates accordingly.
        while True:
            sat = sat_solver.solve()
            if not sat:
                break
            model = sat_solver.get_model()
            if model is None:
                break
            pos = {lit for lit in model if lit > 0}
            selected = [u for u in range(n) if (u + 1) in pos]
            # Compute achieved objective for this selection
            obj_sel = max_dist_of_centers(selected)
            last_sat_solution = selected
            last_sat_obj = obj_sel

            # Shrink candidate list to those strictly less than current objective
            idx = bisect_left(candidates, obj_sel)
            candidates = candidates[:idx]
            if not candidates:
                break
            # Tighten radius: add coverage constraints for the next smaller radius
            add_limit(candidates.pop())

        # Cleanup solver
        try:
            sat_solver.delete()
        except Exception:
            pass

        # Return centers corresponding to the best (smallest) feasible radius found
        return [nodes[i] for i in last_sat_solution]