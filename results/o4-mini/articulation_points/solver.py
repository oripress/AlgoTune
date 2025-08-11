import sys
class Solver:
    def solve(self, problem, **kwargs):
        # allow deep recursion for large graphs
        sys.setrecursionlimit(10**7)
        n = problem["num_nodes"]
        edges = problem["edges"]
        # build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # initialize discovery, low-link, and articulation flags
        disc = [-1] * n
        low = [0] * n
        ap = [False] * n
        time = 0

        def dfs(u, parent):
            nonlocal time
            disc[u] = low[u] = time
            time += 1
            child_count = 0
            for v in adj[u]:
                if disc[v] == -1:
                    child_count += 1
                    dfs(v, u)
                    if low[v] < low[u]:
                        low[u] = low[v]
                    if parent != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            if parent == -1 and child_count > 1:
                ap[u] = True

        for i in range(n):
            if disc[i] == -1:
                dfs(i, -1)

        return {"articulation_points": [i for i, flag in enumerate(ap) if flag]}