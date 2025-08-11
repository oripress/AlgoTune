from typing import Any, Dict, List
import heapq

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> List[int]:
        """
        Solve Dynamic Assortment Planning (DAP).

        Fast heuristics:
          1. If assigning each period its best product respects capacities -> return it.
          2. Greedy by descending edge weights; if its value reaches a cheap upper bound -> return it.

        Otherwise use OR-Tools SimpleMinCostFlow (fast C++). If OR-Tools is unavailable or fails,
        a Python min-cost max-flow (successive shortest augmenting path with potentials) fallback is used.
        """
        T = int(problem.get("T", 0))
        N = int(problem.get("N", 0))
        prices = problem.get("prices", []) or []
        capacities = problem.get("capacities", []) or []
        probs = problem.get("probs", []) or []

        if T <= 0:
            return []
        if N <= 0:
            return [-1] * T

        # Capacities normalized
        caps = [max(0, int(capacities[i]) if i < len(capacities) else 0) for i in range(N)]
        if sum(caps) == 0:
            return [-1] * T

        # Compute rewards W[t][i] = price[i] * probs[t][i]
        W: List[List[float]] = [[0.0] * N for _ in range(T)]
        max_w = 0.0
        per_period_best: List[int] = [0] * T
        per_period_best_val: List[float] = [0.0] * T
        for t in range(T):
            row = probs[t] if t < len(probs) else [0.0] * N
            best_i = 0
            best_v = -1.0
            for i in range(N):
                p = float(prices[i]) if i < len(prices) else 0.0
                pr = float(row[i]) if i < len(row) else 0.0
                w = p * pr
                W[t][i] = w
                if w > best_v:
                    best_v = w
                    best_i = i
                if w > max_w:
                    max_w = w
            per_period_best[t] = best_i
            per_period_best_val[t] = best_v

        # Quick check: assign every period its best product; if capacities allow, this is optimal.
        counts = [0] * N
        feasible_best = True
        for t, bi in enumerate(per_period_best):
            counts[bi] += 1
            if counts[bi] > caps[bi]:
                feasible_best = False
                break
        if feasible_best:
            return [per_period_best[t] for t in range(T)]

        # Greedy heuristic: sort edges by weight and pick if period unused and product has capacity.
        edges: List[tuple] = []
        for t in range(T):
            for i in range(N):
                if caps[i] > 0:
                    w = W[t][i]
                    # Skip zero-weight edges (idle covers those)
                    if w > 0.0:
                        edges.append((w, t, i))
        edges.sort(reverse=True, key=lambda x: x[0])

        chosen = [-1] * T
        used = [0] * N
        selected = 0
        total_caps = min(sum(caps), T)
        for w, t, i in edges:
            if chosen[t] != -1:
                continue
            if used[i] >= caps[i]:
                continue
            chosen[t] = i
            used[i] += 1
            selected += 1
            if selected >= total_caps:
                break

        greedy_rev = sum(W[t][chosen[t]] for t in range(T) if chosen[t] != -1)

        # Upper bounds: UB1 = sum of best per period; UB2 = sum of top cap_i rewards per product
        UB1 = sum(per_period_best_val)
        UB2 = 0.0
        for i in range(N):
            if caps[i] <= 0:
                continue
            col = [W[t][i] for t in range(T)]
            if caps[i] < T:
                col.sort(reverse=True)
                UB2 += sum(col[:caps[i]])
            else:
                UB2 += sum(col)
        ub = min(UB1, UB2)

        # If greedy reaches the upper bound (within tolerance) it's optimal -> return it
        if greedy_rev + 1e-9 >= ub:
            return [chosen[t] if chosen[t] != -1 else -1 for t in range(T)]

        # Otherwise, use OR-Tools SimpleMinCostFlow for exact solution when available.
        try:
            from ortools.graph import pywrapgraph  # type: ignore

            # Node indexing: period nodes 0..T-1, product nodes T..T+N-1, sink = T+N
            period_offset = 0
            prod_offset = T
            SINK = T + N

            mcf = pywrapgraph.SimpleMinCostFlow()

            # Scaling factor to convert float costs to integers. Keep reasonably large to avoid ties.
            SCALE = 100000
            edges_map: List[List[int]] = [[-1] * (N + 1) for _ in range(T)]
            idle_cost_int = int(round(max_w * SCALE))

            # Add arcs: period -> products and period -> sink (idle)
            for t in range(T):
                u = period_offset + t
                for i in range(N):
                    if caps[i] <= 0:
                        continue
                    v = prod_offset + i
                    cost_int = int(round((max_w - W[t][i]) * SCALE))
                    arc_id = mcf.AddArcWithCapacityAndUnitCost(u, v, 1, cost_int)
                    edges_map[t][i] = arc_id
                # idle arc
                arc_idle = mcf.AddArcWithCapacityAndUnitCost(u, SINK, 1, idle_cost_int)
                edges_map[t][N] = arc_idle
                mcf.SetNodeSupply(u, 1)

            # product -> sink arcs
            for i in range(N):
                capi = min(caps[i], T)
                if capi > 0:
                    mcf.AddArcWithCapacityAndUnitCost(prod_offset + i, SINK, capi, 0)
            # sink demand = -T
            mcf.SetNodeSupply(SINK, -T)

            status = mcf.Solve()
            if status == mcf.OPTIMAL:
                result: List[int] = [-1] * T
                for t in range(T):
                    chosen_t = -1
                    for i in range(N + 1):
                        arc_id = edges_map[t][i]
                        if arc_id is not None and arc_id >= 0:
                            if mcf.Flow(arc_id) > 0:
                                chosen_t = -1 if i == N else i
                                break
                    result[t] = chosen_t
                return result
            # If OR-Tools didn't return OPTIMAL, fall through to Python fallback
        except Exception:
            # OR-Tools not available or failed; continue to Python fallback
            pass

        # Python fallback: min-cost max-flow (successive shortest augmenting path with potentials)
        # Build graph: SRC -> period nodes -> product nodes -> SINK
        Np = N + 1  # include idle product
        SRC = 0
        period_offset = 1
        prod_offset = period_offset + T
        SINK = prod_offset + Np
        V = SINK + 1

        class Edge:
            __slots__ = ("to", "rev", "cap", "cost")

            def __init__(self, to: int, rev: int, cap: int, cost: float):
                self.to = int(to)
                self.rev = int(rev)
                self.cap = int(cap)
                self.cost = float(cost)

        graph: List[List[Edge]] = [[] for _ in range(V)]

        def add_edge(u: int, v: int, cap: int, cost: float) -> None:
            graph[u].append(Edge(v, len(graph[v]), cap, cost))
            graph[v].append(Edge(u, len(graph[u]) - 1, 0, -cost))

        # Source -> period nodes
        for t in range(T):
            add_edge(SRC, period_offset + t, 1, 0.0)

        # Period -> product edges; keep refs for decoding
        edges_map_py: List[List[Any]] = [[None] * Np for _ in range(T)]
        C = max_w
        for t in range(T):
            u = period_offset + t
            for i in range(N):
                if caps[i] <= 0:
                    continue
                v = prod_offset + i
                add_edge(u, v, 1, C - W[t][i])
                edges_map_py[t][i] = graph[u][-1]
            # idle
            v_idle = prod_offset + N
            add_edge(u, v_idle, 1, C)
            edges_map_py[t][N] = graph[u][-1]

        # Product -> sink edges
        for i in range(N):
            capi = min(caps[i], T)
            if capi > 0:
                add_edge(prod_offset + i, SINK, capi, 0.0)
        add_edge(prod_offset + N, SINK, T, 0.0)

        INF = 1e30
        potential = [0.0] * V
        dist = [0.0] * V
        prevv = [0] * V
        preve = [0] * V

        flow = 0
        maxflow = T

        while flow < maxflow:
            for v in range(V):
                dist[v] = INF
            dist[SRC] = 0.0
            pq = [(0.0, SRC)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u] + 1e-15:
                    continue
                pu = potential[u]
                for ei, e in enumerate(graph[u]):
                    if e.cap <= 0:
                        continue
                    v = e.to
                    nd = d + e.cost + pu - potential[v]
                    if nd + 1e-15 < dist[v]:
                        dist[v] = nd
                        prevv[v] = u
                        preve[v] = ei
                        heapq.heappush(pq, (nd, v))

            if dist[SINK] >= INF / 2:
                break  # no augmenting path

            for v in range(V):
                if dist[v] < INF / 2:
                    potential[v] += dist[v]

            # bottleneck
            addf = maxflow - flow
            v = SINK
            while v != SRC:
                u = prevv[v]
                e = graph[u][preve[v]]
                if e.cap < addf:
                    addf = e.cap
                v = u

            if addf == 0:
                break

            # augment
            v = SINK
            while v != SRC:
                u = prevv[v]
                e = graph[u][preve[v]]
                e.cap -= addf
                graph[v][e.rev].cap += addf
                v = u

            flow += addf

        # Decode assignment
        result: List[int] = [-1] * T
        for t in range(T):
            for i in range(Np):
                e = edges_map_py[t][i]
                if e is not None and e.cap == 0:
                    result[t] = -1 if i == N else i
                    break

        return result