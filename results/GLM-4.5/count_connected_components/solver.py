class Solver:
    def solve(self, problem, **kwargs):
        try:
            n = problem.get("num_nodes", 0)
            edges = problem.get("edges", [])
            
            if n == 0:
                return {"number_connected_components": 0}
                
            # Build adjacency list using tuples for better memory efficiency
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Convert to tuples for faster iteration
            for i in range(n):
                adj[i] = tuple(adj[i])
            
            visited = [False] * n
            count = 0
            
            # Pre-allocate queue
            queue = [0] * n
            
            for i in range(n):
                if not visited[i]:
                    count += 1
                    queue_start = 0
                    queue_end = 1
                    queue[0] = i
                    visited[i] = True
                    
                    while queue_start < queue_end:
                        node = queue[queue_start]
                        queue_start += 1
                        # Direct iteration over neighbors
                        for neighbor in adj[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                queue[queue_end] = neighbor
                                queue_end += 1
                    
            return {"number_connected_components": count}
            
        except Exception as e:
            return {"number_connected_components": -1}