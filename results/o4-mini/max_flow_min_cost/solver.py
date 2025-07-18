import importlib
import heapq
try:
    from solver_cy import CySolver
except ImportError:
    CySolver = None
# Dynamic import of OR-Tools MCMF if available
SimpleMinCostFlow = None
for module_name in (
    'ortools.graph.pywrapgraph',
    'ortools.graph.min_cost_flow',
    'ortools.min_cost_flow',
    'ortools.pywrapgraph'
):
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, 'SimpleMinCostFlow', None)
        if cls:
            SimpleMinCostFlow = cls
            break
    except ImportError:
        continue

class Solver:
    def solve(self, problem, **kwargs):
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)

        # OR-Tools C++ MCMF
        # OR-Tools C++ MCMF
        if SimpleMinCostFlow:
            try:
                mcmf = SimpleMinCostFlow()
                add_arc = mcmf.AddArcWithCapacityAndUnitCost
                arc_u = []
                arc_v = []
                for i in range(n):
                    cap_i = capacity[i]
                    cost_i = cost[i]
                    for j, cij in enumerate(cap_i):
                        if cij:
                            add_arc(i, j, cij, cost_i[j])
                            arc_u.append(i)
                            arc_v.append(j)
                status = mcmf.SolveMaxFlowWithMinCost(s, t)
                if status == mcmf.OPTIMAL:
                    solution = [[0] * n for _ in range(n)]
                    sol = solution
                    au = arc_u
                    av = arc_v
                    flow = mcmf.Flow
                    for idx in range(len(au)):
                        f = flow(idx)
                        if f:
                            sol[au[idx]][av[idx]] = f
                    return solution
            except Exception:
                pass
        # Python fallback: successive shortest path
        u = []
        v = []
        cap_list = []
        cost_list = []
        adj = [[] for _ in range(n)]
        orig = [[-1] * n for _ in range(n)]
        # Build residual graph
        for i, row_cap in enumerate(capacity):
            for j, cij in enumerate(row_cap):
                if cij:
                    # forward edge
                    idx = len(u)
                    u.append(i); v.append(j)
                    cap_list.append(cij); cost_list.append(cost[i][j])
                    adj[i].append(idx)
                    # reverse edge
                    ridx = len(u)
                    u.append(j); v.append(i)
                    cap_list.append(0); cost_list.append(-cost[i][j])
                    adj[j].append(ridx)
                    orig[i][j] = idx

        INF = 10**30
        potential = [0] * n
        heappush = heapq.heappush
        heappop = heapq.heappop

        # Successive Shortest Path algorithm
        while True:
            dist = [INF] * n
            prev_node = [-1] * n
            prev_e = [-1] * n
            dist[s] = 0
            hq = [(0, s)]
            while hq:
                d, node = heappop(hq)
                if d > dist[node]:
                    continue
                pot_node = potential[node]
                for eid in adj[node]:
                    if cap_list[eid] <= 0:
                        continue
                    to = v[eid]
                    nd = d + cost_list[eid] + pot_node - potential[to]
                    if nd < dist[to]:
                        dist[to] = nd
                        prev_node[to] = node
                        prev_e[to] = eid
                        heappush(hq, (nd, to))
            # no more augmenting path
            if prev_node[t] == -1:
                break
            # update potentials
            for i in range(n):
                if dist[i] < INF:
                    potential[i] += dist[i]
            # find bottleneck
            push = INF
            node = t
            while node != s:
                eid = prev_e[node]
                if cap_list[eid] < push:
                    push = cap_list[eid]
                node = u[eid]
            # apply flow
            node = t
            while node != s:
                eid = prev_e[node]
                cap_list[eid] -= push
                cap_list[eid ^ 1] += push
                node = u[eid]

        # Build solution from residual caps
        solution = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                eid = orig[i][j]
                if eid != -1:
                    solution[i][j] = capacity[i][j] - cap_list[eid]
        return solution