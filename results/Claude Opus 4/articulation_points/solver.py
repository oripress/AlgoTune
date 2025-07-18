from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[int]]:
        """Find articulation points using Tarjan's algorithm."""
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        # Build adjacency list
        adj = [[] for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Tarjan's algorithm
        visited = [False] * num_nodes
        disc = [0] * num_nodes
        low = [0] * num_nodes
        parent = [-1] * num_nodes
        articulation_points = set()
        time = [0]
        
        def dfs(u):
            children = 0
            visited[u] = True
            disc[u] = low[u] = time[0]
            time[0] += 1
            
            for v in adj[u]:
                if not visited[v]:
                    children += 1
                    parent[v] = u
                    dfs(v)
                    
                    low[u] = min(low[u], low[v])
                    
                    # u is an articulation point if:
                    # 1. u is root and has more than one child
                    if parent[u] == -1 and children > 1:
                        articulation_points.add(u)
                    
                    # 2. u is not root and low[v] >= disc[u]
                    if parent[u] != -1 and low[v] >= disc[u]:
                        articulation_points.add(u)
                        
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
        
        # Run DFS from all unvisited nodes
        for i in range(num_nodes):
            if not visited[i]:
                dfs(i)
        
        return {"articulation_points": sorted(list(articulation_points))}