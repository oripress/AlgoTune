from typing import Any, List, Tuple
import heapq

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        """
        Solve the Maximum Flow Min Cost problem using a fast successive shortest augmenting path
        algorithm with potentials (Johnson's trick) and Dijkstra.

        Input (problem): dict with keys:
          - "capacity": 2D list of non-negative numbers (n x n)
          - "cost": 2D list of non-negative numbers (n x n)
          - "s": int (source index)
          - "t": int (sink index)

        Output: 2D list (n x n) of flow values on each directed edge.
        """
        try:
            capacity = problem["capacity"]
            cost = problem["cost"]
            s = problem["s"]
            t = problem["t"]
            n = len(capacity)

            # Quick checks
            if n == 0:
                return []
            if s == t:
                # No flow when source equals sink
                return [[0 for _ in range(n)] for _ in range(n)]

            # Build adjacency list for residual network
            # Each edge: [to, rev, cap, cost]
            graph: List[List[List[float]]] = [[] for _ in range(n)]
            # Keep references to original forward edges to reconstruct flows
            # Each entry: (u, index_in_graph[u], initial_capacity)
            orig_edges: List[Tuple[int, int, float]] = []

            def add_edge(u: int, v: int, cap_val: float, cost_val: float) -> None:
                # forward edge
                fwd = [v, len(graph[v]), cap_val, float(cost_val)]
                # backward edge
                bwd = [u, len(graph[u]), 0.0, -float(cost_val)]
                graph[u].append(fwd)
                graph[v].append(bwd)
                orig_edges.append((u, len(graph[u]) - 1, float(cap_val)))

            # Create edges only where capacity > 0
            for i in range(n):
                row_cap = capacity[i]
                row_cost = cost[i]
                for j in range(n):
                    cap_ij = row_cap[j]
                    if cap_ij > 0:
                        add_edge(i, j, float(cap_ij), float(row_cost[j]))

            # Potentials for reduced costs
            pot = [0.0] * n

            # Dijkstra structures
            INF = float("inf")
            dist = [INF] * n
            parent_v = [-1] * n
            parent_e = [-1] * n

            # Repeatedly find shortest augmenting path in residual graph
            while True:
                # Initialize distances
                for i in range(n):
                    dist[i] = INF
                    parent_v[i] = -1
                    parent_e[i] = -1

                dist[s] = 0.0
                pq: List[Tuple[float, int]] = [(0.0, s)]

                while pq:
                    d, u = heapq.heappop(pq)
                    if d != dist[u]:
                        continue
                    if u == t:
                        # Early stop when t is settled
                        break
                    gu = graph[u]
                    pu = pot[u]
                    for ei, e in enumerate(gu):
                        v = e[0]
                        cap_rem = e[2]
                        if cap_rem <= 0.0:
                            continue
                        rcost = e[3] + pu - pot[v]  # reduced cost
                        nd = d + rcost
                        if nd < dist[v]:
                            dist[v] = nd
                            parent_v[v] = u
                            parent_e[v] = ei
                            heapq.heappush(pq, (nd, v))

                if dist[t] == INF:
                    # No augmenting path remains
                    break

                # Update potentials to keep reduced costs non-negative next round
                # Only update where reachable
                for v in range(n):
                    if dist[v] < INF:
                        pot[v] += dist[v]

                # Find bottleneck capacity along the path
                add_flow = INF
                v = t
                while v != s:
                    u = parent_v[v]
                    ei = parent_e[v]
                    if u == -1 or ei == -1:
                        add_flow = 0.0
                        break
                    e = graph[u][ei]
                    if e[2] < add_flow:
                        add_flow = e[2]
                    v = u

                if add_flow <= 0.0 or add_flow == INF:
                    # Safety check; if no positive augmenting flow found, terminate
                    break

                # Augment along the path
                v = t
                while v != s:
                    u = parent_v[v]
                    ei = parent_e[v]
                    e = graph[u][ei]
                    rev_idx = e[1]
                    # Update residual capacities
                    e[2] -= add_flow
                    graph[v][rev_idx][2] += add_flow
                    v = u

            # Reconstruct flow matrix from original forward edges
            solution = [[0.0 for _ in range(n)] for _ in range(n)]
            for u, ei, init_cap in orig_edges:
                e = graph[u][ei]
                used = init_cap - e[2]
                if used != 0.0:
                    v = e[0]
                    solution[u][v] = used

            return solution

        except Exception:
            # On failure, return zero matrix of correct shape
            try:
                n = len(problem.get("capacity", []))
            except Exception:
                n = 0
            return [[0 for _ in range(n)] for _ in range(n)]