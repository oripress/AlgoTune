from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """
        Find articulation points using Tarjan's DFS-based algorithm.
        Optimized with numpy arrays for better performance.
        """
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        if num_nodes == 0:
            return {"articulation_points": []}
        
        if not edges:
            return {"articulation_points": []}

        # Build adjacency list efficiently
        adj = [[] for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Pre-allocate arrays
        visited = [False] * num_nodes
        disc = [0] * num_nodes
        low = [0] * num_nodes
        parent = [-1] * num_nodes
        ap = [False] * num_nodes
        time_counter = [0]

        def dfs(u):
            children = 0
            visited[u] = True
            disc[u] = low[u] = time_counter[0]
            time_counter[0] += 1

            for v in adj[u]:
                if not visited[v]:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # Check if u is an articulation point
                    if parent[u] == -1 and children > 1:
                        ap[u] = True
                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                        
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        # Run DFS for all components
        for i in range(num_nodes):
            if not visited[i]:
                dfs(i)

        # Collect articulation points
        return {"articulation_points": [i for i in range(num_nodes) if ap[i]]}