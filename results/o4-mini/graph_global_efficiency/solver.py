import math
from collections import deque

class Solver:
    def solve(self, problem, **kwargs):
        adj = problem.get("adjacency_list", [])
        n = len(adj)
        if n <= 1:
            return {"global_efficiency": 0.0}
        total_inv = 0.0
        # BFS from each node
        for u in range(n):
            # distance initialization
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                v = q.popleft()
                for w in adj[v]:
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        q.append(w)
            # sum inverse distances for ordered pairs
            for v2 in range(n):
                d = dist[v2]
                if v2 != u and d > 0:
                    total_inv += 1.0 / d
        eff = total_inv / (n * (n - 1))
        return {"global_efficiency": eff}