from __future__ import annotations

from numbers import Integral
from typing import Any
import heapq
import math

class Solver:
    def __init__(self) -> None:
        self._SimpleMaxFlow = None
        self._SimpleMinCostFlow = None
        try:
            from ortools.graph.python import max_flow as _max_flow
            from ortools.graph.python import min_cost_flow as _min_cost_flow

            self._SimpleMaxFlow = _max_flow.SimpleMaxFlow
            self._SimpleMinCostFlow = _min_cost_flow.SimpleMinCostFlow
        except Exception:
            try:
                from ortools.graph import pywrapgraph as _pywrapgraph

                self._SimpleMaxFlow = _pywrapgraph.SimpleMaxFlow
                self._SimpleMinCostFlow = _pywrapgraph.SimpleMinCostFlow
            except Exception:
                pass

    @staticmethod
    def _zero_matrix(n: int) -> list[list[Any]]:
        return [[0 for _ in range(n)] for _ in range(n)]

    @staticmethod
    def _is_int_valued(x: Any) -> bool:
        if isinstance(x, Integral):
            return True
        try:
            return float(x).is_integer()
        except Exception:
            return False

    @staticmethod
    def _method(obj: Any, snake: str, camel: str):
        fn = getattr(obj, snake, None)
        return fn if fn is not None else getattr(obj, camel)

    def _solve_ortools(
        self, n: int, edges: list[tuple[int, int, Any, Any]], s: int, t: int
    ) -> list[list[Any]] | None:
        if self._SimpleMinCostFlow is None:
            return None

        int_edges: list[tuple[int, int, int, int]] = []
        total_out_s = 0
        total_in_t = 0
        max_cost = 0

        for u, v, c, w in edges:
            ci = int(c)
            wi = int(w)
            int_edges.append((u, v, ci, wi))
            if u == s:
                total_out_s += ci
            if v == t:
                total_in_t += ci
            if wi > max_cost:
                max_cost = wi

        maxflow_cap = total_out_s if total_out_s < total_in_t else total_in_t
        sol = self._zero_matrix(n)
        if maxflow_cap <= 0:
            return sol

        base = (n + 1) * max_cost + 1
        penalty = n * base + 1

        mcf = self._SimpleMinCostFlow()
        add_arc_cost = self._method(
            mcf,
            "add_arc_with_capacity_and_unit_cost",
            "AddArcWithCapacityAndUnitCost",
        )
        set_supply = self._method(mcf, "set_node_supply", "SetNodeSupply")
        solve_mcf = self._method(mcf, "solve", "Solve")
        num_arcs = self._method(mcf, "num_arcs", "NumArcs")
        flow = self._method(mcf, "flow", "Flow")
        tail = self._method(mcf, "tail", "Tail")
        head = self._method(mcf, "head", "Head")

        for u, v, c, w in int_edges:
            add_arc_cost(u, v, c, w * (n + 1) + 1)
        artificial_idx = len(int_edges)
        add_arc_cost(t, s, maxflow_cap, -penalty)

        for i in range(n):
            set_supply(i, 0)

        status = solve_mcf()
        optimal = getattr(mcf, "OPTIMAL", None)
        if optimal is not None and status != optimal:
            return None

        for i in range(num_arcs()):
            if i == artificial_idx:
                continue
            f = flow(i)
            if f:
                sol[tail(i)][head(i)] = f
        return sol

    def _solve_ssp(
        self, n: int, edges: list[tuple[int, int, Any, Any]], s: int, t: int
    ) -> list[list[Any]]:
        sol = self._zero_matrix(n)
        if not edges or s == t:
            return sol

        adj: list[list[int]] = [[] for _ in range(n)]
        to: list[int] = []
        cap: list[Any] = []
        cost: list[Any] = []
        rev: list[int] = []
        forwards: list[tuple[int, int, int]] = []

        for u, v, c, w in edges:
            idx = len(to)

            adj[u].append(idx)
            to.append(v)
            cap.append(c)
            cost.append(w)
            rev.append(idx + 1)

            adj[v].append(idx + 1)
            to.append(u)
            cap.append(0)
            cost.append(-w)
            rev.append(idx)

            forwards.append((u, v, idx))

        inf = math.inf
        pot = [0] * n
        dist = [inf] * n
        parent = [-1] * n
        heappush = heapq.heappush
        heappop = heapq.heappop
        eps = 1e-12

        while True:
            for i in range(n):
                dist[i] = inf
                parent[i] = -1
            dist[s] = 0
            heap: list[tuple[Any, int]] = [(0, s)]

            while heap:
                d, u = heappop(heap)
                if d > dist[u]:
                    continue
                pu = pot[u]
                for ei in adj[u]:
                    residual = cap[ei]
                    if residual <= eps:
                        continue
                    v = to[ei]
                    nd = d + cost[ei] + pu - pot[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        parent[v] = ei
                        heappush(heap, (nd, v))

            if parent[t] == -1:
                break

            for i in range(n):
                if dist[i] < inf:
                    pot[i] += dist[i]

            pushed = inf
            v = t
            while v != s:
                ei = parent[v]
                if cap[ei] < pushed:
                    pushed = cap[ei]
                v = to[rev[ei]]

            if pushed <= eps:
                break

            v = t
            while v != s:
                ei = parent[v]
                rei = rev[ei]
                cap[ei] -= pushed
                cap[rei] += pushed
                v = to[rei]

        for u, v, idx in forwards:
            f = cap[rev[idx]]
            if f:
                sol[u][v] = f
        return sol

    def solve(self, problem, **kwargs) -> Any:
        try:
            capacity = problem["capacity"]
            costs = problem["cost"]
            s = problem["s"]
            t = problem["t"]
            n = len(capacity)

            if n == 0 or s == t:
                return self._zero_matrix(n)

            edges: list[tuple[int, int, Any, Any]] = []
            all_integral = True
            for i, row in enumerate(capacity):
                for j, c in enumerate(row):
                    if c > 0:
                        w = costs[i][j]
                        edges.append((i, j, c, w))
                        if all_integral and (
                            not self._is_int_valued(c) or not self._is_int_valued(w)
                        ):
                            all_integral = False

            if not edges:
                return self._zero_matrix(n)

            if all_integral:
                try:
                    ans = self._solve_ortools(n, edges, s, t)
                    if ans is not None:
                        return ans
                except Exception:
                    pass

            return self._solve_ssp(n, edges, s, t)
        except Exception:
            try:
                n = len(problem["capacity"])
            except Exception:
                n = 0
            return self._zero_matrix(n)