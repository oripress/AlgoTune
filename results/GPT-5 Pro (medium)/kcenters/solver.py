from __future__ import annotations

from typing import Any, Dict, List, Tuple
import heapq
import math
import bisect

# We rely on python-sat for exact feasibility checking
from pysat.solvers import Solver as SATSolver
from pysat.formula import CNF
from pysat.card import CardEnc, EncType

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Vertex K-Center problem exactly using:
        - All-pairs shortest paths (via Dijkstra from each node)
        - Binary search on candidate radii
        - SAT-based feasibility check for coverage with at most k centers

        Input:
            problem: tuple (G, k)
                G: dict[str, dict[str, float]] adjacency with symmetric weights
                k: int, number of centers allowed

        Output:
            list[str]: identifiers of chosen centers (length <= k)
        """
        G_dict, k = problem
        # Nodes and indexing
        nodes: List[str] = list(G_dict.keys())
        n = len(nodes)

        # Handle trivial cases early
        if n == 0:
            return []
        if k >= n:
            return nodes[:k]
        if k <= 0:
            return []

        idx_of: Dict[str, int] = {v: i for i, v in enumerate(nodes)}

        # Build adjacency as index-based list
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for u, nbrs in G_dict.items():
            ui = idx_of[u]
            for v, w in nbrs.items():
                vi = idx_of[v]
                adj[ui].append((vi, float(w)))

        # Dijkstra from each node to compute all-pairs shortest paths
        def dijkstra(src: int) -> List[float]:
            dist = [math.inf] * n
            dist[src] = 0.0
            heap: List[Tuple[float, int]] = [(0.0, src)]
            while heap:
                du, u = heapq.heappop(heap)
                if du != dist[u]:
                    continue
                for v, w in adj[u]:
                    nd = du + w
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
            return dist

        all_dists: List[List[float]] = [dijkstra(i) for i in range(n)]

        # Prepare sorted unique candidate radii (pairwise distances)
        cand_set = set()
        for i in range(n):
            for j in range(n):
                d = all_dists[i][j]
                if d != math.inf:
                    cand_set.add(d)
        if not cand_set:
            # No finite distances? Then any single node covers itself with 0 radius.
            return nodes[: min(k, n)]
        cand_dists = sorted(cand_set)

        # Pre-sort per-row distances and nodes for fast threshold lookup
        row_sorted_dists: List[List[float]] = []
        row_sorted_nodes: List[List[int]] = []
        for v in range(n):
            pairs = sorted((all_dists[v][u], u) for u in range(n))
            row_sorted_dists.append([p[0] for p in pairs])
            row_sorted_nodes.append([p[1] for p in pairs])

        # Greedy farthest-first heuristic (Gonzalez) to get an initial feasible solution and radius bound
        centers: List[int] = []
        # First center: node minimizing its eccentricity (as in reference)
        first_center = min(range(n), key=lambda c: max(all_dists[c][u] for u in range(n)))
        centers.append(first_center)

        # Pick remaining centers
        assigned_min = [all_dists[first_center][v] for v in range(n)]
        while len(centers) < k:
            next_center = max(range(n), key=lambda v: assigned_min[v])
            centers.append(next_center)
            dv = all_dists[next_center]
            for v in range(n):
                if dv[v] < assigned_min[v]:
                    assigned_min[v] = dv[v]

        # Current objective (upper bound on optimal radius)
        ub = max(assigned_min)

        # Find index range for binary search (only distances <= ub)
        hi = bisect.bisect_right(cand_dists, ub) - 1
        if hi < 0:
            hi = 0
        lo = 0

        # SAT Feasibility checker
        # Given radius r, returns selected centers indices if feasible, else None
        def check_feasible(r: float) -> List[int] | None:
            # Build covering clauses
            cnf = CNF()
            var_ids = [i + 1 for i in range(n)]

            # Coverage constraints: for each node v, it must be within r of some chosen center u
            for v in range(n):
                ds = row_sorted_dists[v]
                pos = bisect.bisect_right(ds, r)
                if pos <= 0:
                    # No one can cover v at radius r -> infeasible
                    return None
                cover_nodes = row_sorted_nodes[v][:pos]
                clause = [var_ids[u] for u in cover_nodes]
                cnf.append(clause)

            # At most k centers
            card = CardEnc.atmost(lits=var_ids, bound=k, encoding=EncType.totalizer)
            cnf.extend(card.clauses)

            # Solve CNF
            with SATSolver(name="mc", bootstrap_with=cnf.clauses) as s:
                sat = s.solve()
                if not sat:
                    return None
                model = s.get_model()
                if model is None:
                    return None
                model_set = set(model)
                chosen = [i for i, v in enumerate(var_ids) if v in model_set]
                if len(chosen) > k:
                    chosen = chosen[:k]
                return chosen

        best_centers_idx: List[int] | None = None

        # Binary search for minimal feasible radius
        while lo < hi:
            mid = (lo + hi) // 2
            r = cand_dists[mid]
            sol = check_feasible(r)
            if sol is not None:
                best_centers_idx = sol
                hi = mid
            else:
                lo = mid + 1

        # Final feasibility at lo
        if best_centers_idx is None:
            final_sol = check_feasible(cand_dists[lo])
            if final_sol is None:
                # Fallback to heuristic centers (shouldn't happen)
                final_sol = centers
            best_centers_idx = final_sol

        # Map indices back to node names
        result = [nodes[i] for i in best_centers_idx[:k]]
        return result