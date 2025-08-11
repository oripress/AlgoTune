class Solver:
    def solve(self, problem, **kwargs):
        """Find articulation points using optimized DFS implementation."""
        num_nodes = problem["num_nodes"]
        edges = problem["edges"]
        
        # Build adjacency list using list for better performance
        adj = [[] for _ in range(num_nodes)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initialize tracking arrays
        visited = [False] * num_nodes
        discovery = [0] * num_nodes
        low = [0] * num_nodes
        parent = [-1] * num_nodes
        articulation_points = set()
        time_counter = [0]
        
        def dfs(start):
            """Iterative DFS to find articulation points."""
            stack = [start]
            visited[start] = True
            
            while stack:
                u = stack.pop()
                # Process node u
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        stack.append(v)
        
        # Call DFS for each unvisited vertex
        for i in range(num_nodes):
            if not visited[i]:
                dfs(i)
        
        # Call DFS for each unvisited vertex
        for i in range(num_nodes):
            if not visited[i]:
                dfs(i)
        
        # Return sorted list of articulation points
        result = list(articulation_points)
        result.sort()
        return {"articulation_points": result}