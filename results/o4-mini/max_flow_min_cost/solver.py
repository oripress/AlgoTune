from typing import Any, List
import heapq

def _mcmf(capacity: List[List[Any]], cost: List[List[Any]], s: int, t: int) -> List[List[Any]]:
    n = len(capacity)
    # build adjacency arrays
    head = [-1] * n
    to = []
    cap = []
    cost_arr = []
    rev = []
    next_arr = []
    orig = []
    for u in range(n):
        cu = capacity[u]
        costu = cost[u]
        for v, c in enumerate(cu):
            if c:
                eid = len(to)
                to.append(v)
                cap.append(c)
                cost_arr.append(costu[v])
                rev.append(None)
                next_arr.append(head[u])
                head[u] = eid
                rid = len(to)
                to.append(u)
                cap.append(0)
                cost_arr.append(-costu[v])
                rev.append(eid)
                next_arr.append(head[v])
                head[v] = rid
                rev[eid] = rid
                orig.append((u, eid, c))
    INF = 10**18
    pi = [0] * n
    prevv = [0] * n
    preve = [0] * n
    # successive shortest augmentations
    while True:
        dist = [INF] * n
        dist[s] = 0
        hq = [(0, s)]
        while hq:
            d, u = heapq.heappop(hq)
            if d != dist[u]:
                continue
            eid = head[u]
            while eid != -1:
                if cap[eid] > 0:
                    v0 = to[eid]
                    nd = d + cost_arr[eid] + pi[u] - pi[v0]
                    if nd < dist[v0]:
                        dist[v0] = nd
                        prevv[v0] = u
                        preve[v0] = eid
                        heapq.heappush(hq, (nd, v0))
                eid = next_arr[eid]
        if dist[t] == INF:
            break
        for v0 in range(n):
            if dist[v0] < INF:
                pi[v0] += dist[v0]
        # find bottleneck
        f = INF
        v0 = t
        while v0 != s:
            eid = preve[v0]
            if cap[eid] < f:
                f = cap[eid]
            v0 = prevv[v0]
        # apply augmentation
        v0 = t
        while v0 != s:
            eid = preve[v0]
            cap[eid] -= f
            cap[rev[eid]] += f
            v0 = prevv[v0]
    # extract flow matrix
    res = [[0] * n for _ in range(n)]
    for u, eid, origc in orig:
        used = origc - cap[eid]
        if used:
            res[u][to[eid]] = used
    return res

# OR-Tools two-phase max-flow min-cost solver
try:
    from ortools.graph import pywrapgraph
    _HAVE_ORTOOLS = True
except ImportError:
    _HAVE_ORTOOLS = False

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[List[Any]]:
        capacity = problem.get("capacity", [])
        cost = problem.get("cost", [])
        n = len(capacity)
        if n == 0:
            return []
        s = problem.get("s", 0)
        t = problem.get("t", n - 1)
        if _HAVE_ORTOOLS:
            # Phase 1: max flow
            max_flow_solver = pywrapgraph.SimpleMaxFlow()
            for u in range(n):
                for v in range(n):
                    cap_uv = capacity[u][v]
                    if cap_uv:
                        max_flow_solver.AddArcWithCapacity(u, v, cap_uv)
            status1 = max_flow_solver.Solve(s, t)
            if status1 == max_flow_solver.OPTIMAL:
                flow_val = max_flow_solver.OptimalFlow()
                if flow_val > 0:
                    # Phase 2: min cost flow
                    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
                    for u in range(n):
                        for v in range(n):
                            cap_uv = capacity[u][v]
                            if cap_uv:
                                min_cost_flow.AddArcWithCapacityAndUnitCost(u, v, cap_uv, cost[u][v])
                    min_cost_flow.SetNodeSupply(s, flow_val)
                    min_cost_flow.SetNodeSupply(t, -flow_val)
                    status2 = min_cost_flow.Solve()
                    if status2 == min_cost_flow.OPTIMAL:
                        res = [[0] * n for _ in range(n)]
                        for i in range(min_cost_flow.NumArcs()):
                            u0 = min_cost_flow.Tail(i)
                            v0 = min_cost_flow.Head(i)
                            f0 = min_cost_flow.Flow(i)
                            if f0:
                                res[u0][v0] = f0
                        return res
            # fallback to python if OR-Tools fails
            return _mcmf(capacity, cost, s, t)
        else:
            return _mcmf(capacity, cost, s, t)