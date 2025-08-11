class Solver:
    def solve(self, problem, **kwargs):
        try:
            n = problem.get("num_nodes", 0)
            edges = problem.get("edges", [])
            
            if n == 0:
                return {"number_connected_components": 0}
            
            # Build adjacency list
            adj = [[] for _ in range(n)]
            append = list.append  # Cache method lookup
            for u, v in edges:
                append(adj[u], v)
                append(adj[v], u)
            
            visited = bytearray(n)  # More memory efficient
            visited = bytearray(n)
            components = 0
            unvisited_count = n
            for i in range(n):
                if not visited[i]:
                    components += 1
                    stack = [i]
                    visited[i] = 1
                    unvisited_count -= 1
                    # Process stack more efficiently
                    pop = stack.pop
                    while stack and unvisited_count > 0:
                        node = pop()
                        # Cache method lookup
                        adj_node = adj[node]
                        for neighbor in adj_node:
                            if not visited[neighbor]:
                                visited[neighbor] = 1
                                unvisited_count -= 1
                                stack.append(neighbor)
                    if unvisited_count == 0:
                        break
            
            return {"number_connected_components": components}
            
        except Exception:
            return {"number_connected_components": -1}