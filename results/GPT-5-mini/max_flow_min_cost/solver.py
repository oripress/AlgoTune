from typing import Any, List, Tuple
import heapq

class Solver:
    def solve(self, problem: dict, **kwargs) -> Any:
        """
        Min-Cost Max-Flow using successive shortest path with potentials.
        Expects problem to contain "capacity", "cost", "s", "t" where capacity and cost
        are n x n matrices (lists or array-like). Returns n x n flow matrix.
        """
        try:
            capacity = problem["capacity"]
            cost = problem["cost"]
            s = int(problem["s"])
            t = int(problem["t"])
            n = len(capacity)
            if n == 0:
                return []

            # Edge structure
            class Edge:
                __slots__ = ("v", "cap", "cost", "rev")
                def __init__(self, v, cap, cost, rev):
                    self.v = v
                    self.cap = float(cap)
                    self.cost = float(cost)
                    self.rev = rev

            # build graph adjacency list and list of original edges for output extraction
            adj: List[List[Edge]] = [[] for _ in range(n)]
            orig_edges: List[Tuple[int, int, float, int]] = []  # (u, v, orig_cap, index_in_adj_u)

            for i in range(n):
                row_cap = capacity[i]
                row_cost = cost[i]
                for j in range(n):
                    cap_ij = row_cap[j]
                    # only add positive-capacity original edges
                    try:
                        if cap_ij <= 0:
                            continue
                    except Exception:
                        if not cap_ij:
                            continue
                    f_idx = len(adj[i])
                    r_idx = len(adj[j])
                    adj[i].append(Edge(j, cap_ij, row_cost[j], r_idx))
                    adj[j].append(Edge(i, 0.0, -float(row_cost[j]), f_idx))
                    orig_edges.append((i, j, float(cap_ij), f_idx))

            INF = float("inf")
            potential = [0.0] * n  # potentials for reduced costs
            total_flow = 0.0
            total_cost = 0.0

            # successive shortest augmenting paths
            while True:
                dist = [INF] * n
                prev_node = [-1] * n
                prev_edge = [-1] * n

                dist[s] = 0.0
                heap = [(0.0, s)]
                while heap:
                    d, u = heapq.heappop(heap)
                    if d > dist[u]:
                        continue
                    pu = potential[u]
                    for ei, e in enumerate(adj[u]):
                        if e.cap <= 0.0:
                            continue
                        v = e.v
                        nd = d + e.cost + pu - potential[v]
                        if nd < dist[v]:
                            dist[v] = nd
                            prev_node[v] = u
                            prev_edge[v] = ei
                            heapq.heappush(heap, (nd, v))

                if dist[t] == INF:
                    break  # no more augmenting path

                # update potentials
                for v in range(n):
                    if dist[v] < INF:
                        potential[v] += dist[v]

                # find maximum augment along the path
                addf = INF
                v = t
                while v != s:
                    u = prev_node[v]
                    if u == -1:
                        addf = 0
                        break
                    e = adj[u][prev_edge[v]]
                    if e.cap < addf:
                        addf = e.cap
                    v = u
                if addf == 0 or addf == INF:
                    break

                # apply augmentation
                v = t
                path_cost = 0.0
                while v != s:
                    u = prev_node[v]
                    ei = prev_edge[v]
                    e = adj[u][ei]
                    e.cap -= addf
                    adj[v][e.rev].cap += addf
                    path_cost += e.cost
                    v = u

                total_flow += addf
                total_cost += addf * path_cost

            # build result flow matrix from original edges
            result: List[List[Any]] = [[0 for _ in range(n)] for _ in range(n)]
            tol = 1e-9
            for (u, v, orig_cap, idx) in orig_edges:
                e = adj[u][idx]
                used = orig_cap - e.cap
                if used < tol:
                    used = 0.0
                if abs(used - round(used)) < 1e-9:
                    used = int(round(used))
                result[u][v] = used

            return result

        except Exception:
            try:
                n = len(problem.get("capacity", []))
                return [[0 for _ in range(n)] for _ in range(n)]
            except Exception:
                return []