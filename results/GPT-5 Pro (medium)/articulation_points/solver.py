from typing import Any, Dict, List

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Find articulation points in an undirected graph using Tarjan's algorithm.

        Input:
          - problem: dict with keys:
              * "num_nodes": int
              * "edges": list of [u, v] with 0 <= u < v < num_nodes
        Output:
          - dict with key "articulation_points": sorted list of articulation points (ints)
        """
        n = int(problem.get("num_nodes", 0))
        edges = problem.get("edges", [])
        if n <= 0:
            return {"articulation_points": []}

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(n)]
        for uv in edges:
            u = int(uv[0])
            v = int(uv[1])
            # assuming 0 <= u < v < n, but be robust
            if u == v:
                continue
            if not (0 <= u < n and 0 <= v < n):
                continue
            adj[u].append(v)
            adj[v].append(u)

        # Tarjan's algorithm
        tin = [-1] * n
        low = [0] * n
        is_ap = [False] * n
        timer = 0

        # Use recursion, but ensure recursion limit is sufficient
        # Avoid importing sys at top-level repeatedly
        import sys

        try:
            # set a safe recursion limit based on graph size
            sys.setrecursionlimit(max(1000000, 2 * n + 100))
        except Exception:
            # If setting recursion limit fails, proceed with default
            pass

        def dfs(u: int, parent: int) -> None:
            nonlocal timer
            tin_u = timer
            low_u = tin_u
            tin[u] = tin_u
            timer += 1

            children = 0
            for v in adj[u]:
                if tin[v] == -1:
                    children += 1
                    dfs(v, u)
                    # update low[u] with low[v]
                    lv = low[v]
                    if lv < low_u:
                        low_u = lv
                    # articulation condition for non-root
                    if parent != -1 and lv >= tin_u:
                        is_ap[u] = True
                elif v != parent:
                    tv = tin[v]
                    if tv < low_u:
                        low_u = tv

            low[u] = low_u
            if parent == -1 and children > 1:
                is_ap[u] = True

        for i in range(n):
            if tin[i] == -1:
                dfs(i, -1)

        res = [i for i, flag in enumerate(is_ap) if flag]
        res.sort()
        return {"articulation_points": res}