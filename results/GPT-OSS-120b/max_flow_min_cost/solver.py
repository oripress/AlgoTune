from __future__ import annotations
from typing import Any, List, Tuple
import heapq

class _Edge:
    __slots__ = ("to", "rev", "cap", "cost")
    def __init__(self, to: int, rev: int, cap: int, cost: int):
        self.to = to          # target node
        self.rev = rev        # index of reverse edge in graph[to]
        self.cap = cap        # residual capacity
        self.cost = cost      # cost per unit flow

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[List[int]]:
        """
        Returns a flow matrix that achieves the maximum possible flow from s to t
        with minimum total cost.

        Parameters
        ----------
        problem : dict
            {
                "capacity": [[...]],   # adjacency matrix of capacities (int)
                "cost":     [[...]],   # adjacency matrix of costs (int, non‑negative)
                "s": int,              # source node index
                "t": int               # sink node index
            }

        Returns
        -------
        flow : list[list[int]]
            Matrix where flow[i][j] is the amount sent on edge i→j.
        """
        capacity: List[List[int]] = problem["capacity"]
        cost: List[List[int]] = problem["cost"]
        s: int = problem["s"]
        t: int = problem["t"]
        n: int = len(capacity)

        # ---------- build residual graph ----------
        graph: List[List[_Edge]] = [[] for _ in range(n)]

        def add_edge(u: int, v: int, cap: int, c: int) -> None:
            """Insert forward and reverse edges."""
            fwd = _Edge(v, len(graph[v]), cap, c)
            rev = _Edge(u, len(graph[u]), 0, -c)
            graph[u].append(fwd)
            graph[v].append(rev)

        for i in range(n):
            row_cap = capacity[i]
            row_cost = cost[i]
            for j in range(n):
                if row_cap[j] > 0:
                    add_edge(i, j, row_cap[j], row_cost[j])

        # ---------- min‑cost max‑flow (successive shortest augmenting path) ----------
        INF = 10 ** 18
        potential = [0] * n          # node potentials for reduced costs
        flow_matrix = [[0] * n for _ in range(n)]

        while True:
            # Dijkstra on reduced costs
            dist = [INF] * n
            prev_v = [-1] * n
            prev_e = [-1] * n
            dist[s] = 0
            heap: List[Tuple[int, int]] = [(0, s)]

            while heap:
                d, v = heapq.heappop(heap)
                if d != dist[v]:
                    continue
                for ei, e in enumerate(graph[v]):
                    if e.cap <= 0:
                        continue
                    ndist = d + e.cost + potential[v] - potential[e.to]
                    if ndist < dist[e.to]:
                        dist[e.to] = ndist
                        prev_v[e.to] = v
                        prev_e[e.to] = ei
                        heapq.heappush(heap, (ndist, e.to))

            if dist[t] == INF:
                break  # no augmenting path

            # update potentials
            for v in range(n):
                if dist[v] < INF:
                    potential[v] += dist[v]

            # find bottleneck capacity
            add_cap = INF
            v = t
            while v != s:
                u = prev_v[v]
                e = graph[u][prev_e[v]]
                if e.cap < add_cap:
                    add_cap = e.cap
                v = u

            # augment flow
            v = t
            while v != s:
                u = prev_v[v]
                ei = prev_e[v]
                e = graph[u][ei]
                rev = graph[v][e.rev]

                e.cap -= add_cap
                rev.cap += add_cap

                # record flow on original forward edges only
                if capacity[u][v] > 0:          # original edge existed
                    flow_matrix[u][v] += add_cap
                elif capacity[v][u] > 0:        # traversed a reverse edge, decrease flow
                    flow_matrix[v][u] -= add_cap
                v = u

        return flow_matrix