from typing import Any, List
import heapq

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[List[int]]:
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = int(problem["s"])
        t = int(problem["t"])
        n = len(capacity)

        # Try fast OR-Tools solution: MaxFlow to get F, then MinCostFlow with supply F
        try:
            from ortools.graph import pywrapgraph

            # Build arc lists
            tails: List[int] = []
            heads: List[int] = []
            caps: List[int] = []
            costs: List[int] = []
            for i in range(n):
                row_cap = capacity[i]
                row_cost = cost[i]
                for j, cij in enumerate(row_cap):
                    if cij:
                        tails.append(i)
                        heads.append(j)
                        caps.append(int(cij))
                        costs.append(int(row_cost[j]))

            # 1) Max flow (fast C++)
            mf = pywrapgraph.SimpleMaxFlow()
            for u, v, c in zip(tails, heads, caps):
                mf.AddArcWithCapacity(u, v, c)
            status = mf.Solve(s, t)
            if status != mf.OPTIMAL:
                raise RuntimeError("MaxFlow not optimal")

            F = mf.OptimalFlow()

            # 2) Min-cost flow with supply F at s and demand F at t
            mcf = pywrapgraph.SimpleMinCostFlow()
            for u, v, c, w in zip(tails, heads, caps, costs):
                mcf.AddArcWithCapacityAndUnitCost(u, v, c, w)

            if F:
                mcf.SetNodeSupply(s, F)
                mcf.SetNodeSupply(t, -F)

            mcf_status = mcf.Solve()
            if mcf_status != mcf.OPTIMAL:
                raise RuntimeError("MinCostFlow not optimal")

            # Build solution matrix
            solution = [[0 for _ in range(n)] for _ in range(n)]
            num_arcs = mcf.NumArcs()
            for a in range(num_arcs):
                u = mcf.Tail(a)
                v = mcf.Head(a)
                f = mcf.Flow(a)
                if f:
                    solution[u][v] = int(f)
            return solution
        except Exception:
            # Fallback to optimized Python successive shortest path with potentials
            return self._solve_sssp(capacity, cost, s, t)

    def _solve_sssp(self, capacity: List[List[int]], cost: List[List[int]], s: int, t: int) -> List[List[int]]:
        n = len(capacity)

        # Parallel adjacency arrays per node
        to: List[List[int]] = [[] for _ in range(n)]
        rev: List[List[int]] = [[] for _ in range(n)]
        cap: List[List[int]] = [[] for _ in range(n)]
        cst: List[List[int]] = [[] for _ in range(n)]
        orig: List[List[int]] = [[] for _ in range(n)]

        def add_edge(u: int, v: int, c: int, w: int) -> None:
            # forward
            to[u].append(v)
            rev[u].append(len(to[v]))
            cap[u].append(c)
            cst[u].append(w)
            orig[u].append(c)
            # backward
            to[v].append(u)
            rev[v].append(len(to[u]) - 1)
            cap[v].append(0)
            cst[v].append(-w)
            orig[v].append(0)

        for i in range(n):
            row_cap = capacity[i]
            row_cost = cost[i]
            for j, cij in enumerate(row_cap):
                if cij:
                    add_edge(i, j, int(cij), int(row_cost[j]))

        INF = 10**18
        potential = [0] * n

        dist = [INF] * n
        parent_v = [-1] * n
        parent_e = [-1] * n
        active_nodes: List[int] = []

        while True:
            # Reset only touched nodes for next Dijkstra
            if active_nodes:
                for v in active_nodes:
                    dist[v] = INF
                    parent_v[v] = -1
                    parent_e[v] = -1
                active_nodes.clear()

            dist[s] = 0
            parent_v[s] = -1
            parent_e[s] = -1
            active_nodes.append(s)

            pq = [(0, s)]
            heappop = heapq.heappop
            heappush = heapq.heappush
            pot = potential

            while pq:
                d_u, u = heappop(pq)
            while pq:
                d_u, u = heappop(pq)
                if d_u != dist[u]:
                    continue
                if u == t:
                    break
                to_u = to[u]
                cap_u = cap[u]
                cst_u = cst[u]
                pu = pot[u]
                for ei in range(len(to_u)):
                    if cap_u[ei] <= 0:
                        continue
                    v = to_u[ei]
                    nd = d_u + cst_u[ei] + pu - pot[v]
                    if nd < dist[v]:
                        if dist[v] == INF:
                            active_nodes.append(v)
                        dist[v] = nd
                        parent_v[v] = u
                        parent_e[v] = ei
                        heappush(pq, (nd, v))

            if dist[t] == INF:
                break
            # Update potentials for reached nodes
            for v in active_nodes:
                if dist[v] < INF:
                    pot[v] += dist[v]

            # Find bottleneck
            add = 10**30
            v = t
            while v != s:
                u = parent_v[v]
                ei = parent_e[v]
                if u == -1 or ei == -1:
                    add = 0
                    break
                cu = cap[u][ei]
                if cu < add:
                    add = cu
                v = u
            if add <= 0 or add == 10**30:
                break

            # Augment
            v = t
            while v != s:
                u = parent_v[v]
                ei = parent_e[v]
                ri = rev[u][ei]
                cap[u][ei] -= add
                cap[v][ri] += add
                v = u

        # Extract flows on original edges
        solution = [[0 for _ in range(n)] for _ in range(n)]
        for u in range(n):
            to_u = to[u]
            cap_u = cap[u]
            orig_u = orig[u]
            for ei in range(len(to_u)):
                oc = orig_u[ei]
                if oc > 0:
                    v = to_u[ei]
                    f = oc - cap_u[ei]
                    if f:
                        solution[u][v] = int(f)
        return solution