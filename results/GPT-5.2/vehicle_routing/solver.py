from __future__ import annotations

from typing import Any, List

import numpy as np
from ortools.sat.python import cp_model

try:
    import numba as nb
except Exception:  # pragma: no cover
    nb = None

if nb is not None:

    @nb.njit(cache=True)
    def _held_karp_tour_int(dist: np.ndarray) -> np.ndarray:
        """
        Exact TSP cycle via Held–Karp DP (int64 costs).
        Returns a tour array of length N+1 starting/ending at node 0.
        """
        n = dist.shape[0]
        size = 1 << n
        full = size - 1
        INF = np.int64(1 << 60)

        dp = np.empty((size, n), dtype=np.int64)
        parent = np.empty((size, n), dtype=np.int16)

        for mask in range(size):
            for j in range(n):
                dp[mask, j] = INF
                parent[mask, j] = np.int16(-1)

        dp[1, 0] = 0

        for mask in range(1, size):
            if (mask & 1) == 0:
                continue
            for i in range(n):
                if (mask & (1 << i)) == 0:
                    continue
                base = dp[mask, i]
                if base >= INF:
                    continue
                for j in range(n):
                    if (mask & (1 << j)) != 0:
                        continue
                    nm = mask | (1 << j)
                    v = base + dist[i, j]
                    if v < dp[nm, j]:
                        dp[nm, j] = v
                        parent[nm, j] = np.int16(i)

        best = INF
        last = 1
        for j in range(1, n):
            v = dp[full, j] + dist[j, 0]
            if v < best:
                best = v
                last = j

        tour = np.empty(n + 1, dtype=np.int32)
        tour[n] = 0
        cur = last
        mask = full

        for pos in range(n - 1, 0, -1):
            tour[pos] = cur
            prev = int(parent[mask, cur])
            mask ^= 1 << cur
            cur = prev
        tour[0] = 0
        return tour

class Solver:
    """
    Exact VRP/mTSP solver (single depot, symmetric distances, exactly K non-empty routes).

    Main idea: solve an expanded TSP:
      - Duplicate the depot K times (K depot-copies).
      - Solve a Hamiltonian cycle on these N=K+(n-1) nodes.
      - Forbid depot-copy -> depot-copy arcs to prevent empty routes.
      - Split the cycle at depot copies to obtain K depot-to-depot routes.

    Implementation:
      - For small N: Numba Held–Karp DP (very fast).
      - Otherwise: CP-SAT AddCircuit model (still much faster than MTZ reference).
    """

    def __init__(self) -> None:
        if nb is not None:
            _ = _held_karp_tour_int(np.array([[0, 1], [1, 0]], dtype=np.int64))

    def solve(self, problem: dict[str, Any], **kwargs: Any) -> List[List[int]]:
        D = problem["D"]
        K = int(problem["K"])
        depot = int(problem["depot"])
        n = len(D)

        if K <= 0:
            return []

        customers = [i for i in range(n) if i != depot]
        m = len(customers)

        # No feasible non-empty routes if there are no customers.
        if m == 0:
            return []
        if K > m:
            return []
        if K == m:
            return [[depot, c, depot] for c in customers]

        N = K + m
        actual = [depot] * K + customers

        # Fast exact DP for small N
        if nb is not None and N <= 16:
            Dnp = np.asarray(D, dtype=np.int64)
            INF = np.int64(1 << 50)
            dist = np.empty((N, N), dtype=np.int64)
            for i in range(N):
                ai = actual[i]
                for j in range(N):
                    if i == j:
                        dist[i, j] = 0
                    elif i < K and j < K:
                        dist[i, j] = INF
                    else:
                        dist[i, j] = Dnp[ai, actual[j]]

            tour = _held_karp_tour_int(dist)

            routes: List[List[int]] = []
            cur_route: List[int] = []
            for node in tour[1:]:
                if node < K:
                    routes.append([depot, *cur_route, depot])
                    cur_route = []
                else:
                    cur_route.append(actual[node])

            if len(routes) == K and all(len(r) >= 3 for r in routes):
                return routes
            # If something went wrong (shouldn't), fall back to CP-SAT.

        # CP-SAT AddCircuit fallback
        model = cp_model.CpModel()

        arcs: list[tuple[int, int, cp_model.IntVar]] = []
        lits: list[cp_model.IntVar] = []
        coeffs: list[int] = []

        for i in range(N):
            ai = actual[i]
            for j in range(N):
                if i == j:
                    continue
                if i < K and j < K:
                    continue  # forbid depot-copy adjacency (prevents empty routes)
                v = model.NewBoolVar("")
                arcs.append((i, j, v))
                lits.append(v)
                coeffs.append(int(D[ai][actual[j]]))

        model.AddCircuit(arcs)
        model.Minimize(cp_model.LinearExpr.WeightedSum(lits, coeffs))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1

        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []

        succ = [-1] * N
        for i, j, v in arcs:
            if solver.Value(v) == 1:
                succ[i] = j

        tour = [0]
        cur = 0
        for _ in range(N):
            cur = succ[cur]
            tour.append(cur)
            if cur == 0:
                break

        routes: List[List[int]] = []
        cur_route = []
        for node in tour[1:]:
            if node < K:
                routes.append([depot, *cur_route, depot])
                cur_route = []
            else:
                cur_route.append(actual[node])

        if len(routes) != K or any(len(r) < 3 for r in routes):
            return []
        return routes