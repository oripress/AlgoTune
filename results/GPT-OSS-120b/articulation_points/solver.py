from typing import Any, List, Dict
import sys
sys.setrecursionlimit(10**6)
class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, List[int]]:
        """
        Find articulation points in an undirected graph using Tarjan's algorithm.

        Parameters
        ----------
        problem : dict
            - "num_nodes": int, number of vertices (0‑indexed)
            - "edges": list of [u, v] pairs, each 0 ≤ u < v < num_nodes

        Returns
        -------
        dict
            {"articulation_points": sorted list of articulation vertices}
        """
        n = problem["num_nodes"]
        edges = problem.get("edges", [])

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Tarjan's DFS
        disc = [-1] * n          # discovery times
        low = [0] * n            # low values
        parent = [-1] * n
        time = 0
        ap_set = set()

        def dfs(u: int) -> None:
            nonlocal time
            disc[u] = low[u] = time
            time += 1
            children = 0

            for v in adj[u]:
                if disc[v] == -1:          # Tree edge
                    parent[v] = u
                    children += 1
                    dfs(v)

                    # Update low value
                    if low[u] > low[v]:
                        low[u] = low[v]

                    # Articulation point conditions
                    if parent[u] == -1 and children > 1:
                        ap_set.add(u)
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap_set.add(u)
                elif v != parent[u]:      # Back edge
                    if low[u] > disc[v]:
                        low[u] = disc[v]

        for i in range(n):
            if disc[i] == -1:
                dfs(i)

        return {"articulation_points": sorted(ap_set)}