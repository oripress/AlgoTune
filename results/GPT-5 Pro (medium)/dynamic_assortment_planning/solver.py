from __future__ import annotations

from typing import Any, List
import heapq

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[int]:
        """
        Solve Dynamic Assortment Planning exactly via a fast min-cost max-flow on a layered network.

        Model:
          - Source -> period nodes (capacity 1, cost 0)
          - Period t -> Product i (capacity 1, cost = - price[i] * probs[t][i])
          - Product i -> Sink (capacity = capacities[i], cost 0)

        We send F = min(T, sum capacities) units of flow to maximize total expected revenue.
        Idle periods correspond to periods with no flow.

        Returns
        -------
        List[int]
            offer[t] in {-1, 0, ..., N-1}
        """

        T: int = int(problem["T"])
        N: int = int(problem["N"])
        prices: List[float] = problem["prices"]
        capacities: List[int] = problem["capacities"]
        probs: List[List[float]] = problem["probs"]

        # Quick exits
        if T <= 0 or N <= 0:
            return [-1] * T

        # Filter out products with zero capacity (they can never be offered)
        prod_map = [i for i in range(N) if capacities[i] > 0]
        M = len(prod_map)
        if M == 0:
            return [-1] * T

        total_cap = sum(capacities[i] for i in prod_map)
        F = min(T, total_cap)
        if F <= 0:
            return [-1] * T

        # Build graph
        # Node indexing:
        #   0 .. T-1            : period nodes
        #   T .. T+M-1          : product nodes (filtered)
        #   T+M                 : source
        #   T+M+1               : sink
        V = T + M + 2
        SRC = T + M
        SNK = T + M + 1

        # Lightweight edge structure
        class Edge:
            __slots__ = ("to", "rev", "cap", "cost")

            def __init__(self, to: int, rev: int, cap: int, cost: float):
                self.to = to
                self.rev = rev
                self.cap = cap
                self.cost = cost

        g: List[List[Edge]] = [[] for _ in range(V)]

        def add_edge(u: int, v: int, cap: int, cost: float) -> None:
            g[u].append(Edge(v, len(g[v]), cap, cost))
            g[v].append(Edge(u, len(g[u]) - 1, 0, -cost))

        # Source -> periods
        for t in range(T):
            add_edge(SRC, t, 1, 0.0)

        # Periods -> products, and track per-product max weight for initial potentials
        pbase = T
        max_w_per_prod = [0.0] * M
        # Build all period->product edges
        for t in range(T):
            pt = probs[t]
            # Small local refs for speed
            for j, i in enumerate(prod_map):
                w = prices[i] * pt[i]
                if w > max_w_per_prod[j]:
                    max_w_per_prod[j] = w
                # cost = -weight (maximize weight via min-cost)
                add_edge(t, pbase + j, 1, -w)

        # Products -> sink
        for j, i in enumerate(prod_map):
            cap = capacities[i]
            if cap > 0:
                add_edge(pbase + j, SNK, cap, 0.0)

        # Min-cost max-flow with Johnson potentials + Dijkstra
        # Initialize potentials to ensure nonnegative reduced costs on the initial graph
        # pi[period]=0; pi[product]=-max_w(product); pi[sink]=min(pi[product]); pi[src]=0
        pi = [0.0] * V
        if M > 0:
            min_pi_prod = 0.0
            for j in range(M):
                pi[pbase + j] = -max_w_per_prod[j]
                if j == 0 or pi[pbase + j] < min_pi_prod:
                    min_pi_prod = pi[pbase + j]
            pi[SNK] = min_pi_prod
        # SRC and period nodes remain at 0.0

        # Dijkstra helpers
        INF = float("inf")
        dist = [INF] * V
        prevnode = [-1] * V
        prevedge = [-1] * V

        flow_sent = 0
        # total_cost not needed for output, but we keep it in case of validation/debug
        # total_cost = 0.0

        # Reusable heap
        heap: list[tuple[float, int]] = []

        # Successive shortest augmenting path
        while flow_sent < F:
            # Dijkstra on reduced costs
            for k in range(V):
                dist[k] = INF
                prevnode[k] = -1
                prevedge[k] = -1
            dist[SRC] = 0.0
            heap.clear()
            heap.append((0.0, SRC))

            while heap:
                d_u, u = heapq.heappop(heap)
                if d_u != dist[u]:
                    continue
                gu = g[u]
                pu = pi[u]
                for ei, e in enumerate(gu):
                    if e.cap <= 0:
                        continue
                    v = e.to
                    rc = e.cost + pu - pi[v]  # reduced cost
                    nd = d_u + rc
                    if nd < dist[v]:
                        dist[v] = nd
                        prevnode[v] = u
                        prevedge[v] = ei
                        heapq.heappush(heap, (nd, v))

            if dist[SNK] == INF:
                # No more augmenting path (shouldn't happen if F <= total_cap)
                break

            # Update potentials
            for v in range(V):
                if dist[v] < INF:
                    pi[v] += dist[v]

            # Augment 1 unit along path
            v = SNK
            pcost = 0.0
            while v != SRC:
                u = prevnode[v]
                ei = prevedge[v]
                e = g[u][ei]
                # Accumulate original cost (not reduced)
                pcost += e.cost
                # Push 1 unit of flow
                e.cap -= 1
                g[v][e.rev].cap += 1
                v = u

            flow_sent += 1
            # total_cost += pcost

        # Reconstruct assignment: for each period, find product with net flow (edge cap==0)
        offer = [-1] * T
        for t in range(T):
            for e in g[t]:
                v = e.to
                # Only consider edges from period to product nodes; used iff forward cap == 0
                if pbase <= v < pbase + M and e.cap == 0:
                    offer[t] = prod_map[v - pbase]
                    break

        return offer