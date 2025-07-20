class Solver:
    def solve(self, problem, **kwargs):
        n = problem.get("num_nodes", 0)
        edges = problem.get("edges", [])
        if n <= 0 or not edges:
            return {"articulation_points": []}
        # build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # init arrays
        disc = [-1] * n
        low = [0] * n
        parent = [-1] * n
        ap = [False] * n
        time = [0]
        # DFS with all arrays as default args -> locals in closure
        def dfs(u, adj=adj, disc=disc, low=low, parent=parent, ap=ap, time=time):
            disc[u] = low[u] = time[0]
            time[0] += 1
            child = 0
            for v in adj[u]:
                if disc[v] == -1:
                    parent[v] = u
                    child += 1
                    dfs(v)
                    # update low-link
                    if low[v] < low[u]:
                        low[u] = low[v]
                    # articulation check
                    if parent[u] == -1:
                        if child > 1:
                            ap[u] = True
                    elif low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u] and disc[v] < low[u]:
                    low[u] = disc[v]
        # run dfs from each unvisited node
        for u in range(n):
            if disc[u] == -1:
                dfs(u)
        # collect results
        result = [i for i, flag in enumerate(ap) if flag]
        return {"articulation_points": result}