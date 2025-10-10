from ortools.sat.python import cp_model
import networkx as nx

class Solver:
    def solve(self, problem: list[list[int]]) -> list[int]:
        """
        Optimized solver with special handling for various graph types.
        """
        n = len(problem)
        if n == 0:
            return []
        
        # Build adjacency list for faster lookups
        adj = [[] for _ in range(n)]
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    adj[i].append(j)
                    adj[j].append(i)
                    edge_count += 1
        
        degrees = [len(adj[i]) for i in range(n)]
        
        # All isolated vertices
        if edge_count == 0:
            return list(range(n))
        
        # Handle isolated vertices separately
        isolated = [i for i in range(n) if degrees[i] == 0]
        if isolated:
            non_isolated = [i for i in range(n) if degrees[i] > 0]
            if not non_isolated:
                return list(range(n))
            
            # Solve for non-isolated vertices
            sub_problem = [[problem[v1][v2] for v2 in non_isolated] for v1 in non_isolated]
            sub_solution = self._solve_graph(sub_problem, len(non_isolated))
            return sorted(isolated + [non_isolated[i] for i in sub_solution])
        
        # Check if it's a tree (n-1 edges, connected)
        if edge_count == n - 1 and self._is_connected(adj, n):
            return self._solve_tree(adj, n)
        
        # Check if it's a clique or near-clique
        if edge_count >= n * (n - 1) // 2 - 2:
            # Nearly complete graph - maximum independent set is small
            if edge_count == n * (n - 1) // 2:
                return [0]  # Complete graph
        
        return self._solve_graph(problem, n)
    
    def _is_connected(self, adj, n):
        """Check if graph is connected using BFS."""
        visited = [False] * n
        queue = [0]
        visited[0] = True
        count = 1
        
        while queue:
            v = queue.pop(0)
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    count += 1
                    queue.append(u)
        
        return count == n
    
    def _solve_tree(self, adj, n):
        """Solve maximum independent set for a tree using DP."""
        # Find a root (any node)
        root = 0
        
        # DP on tree
        # dp[v][0] = max independent set size in subtree of v, not including v
        # dp[v][1] = max independent set size in subtree of v, including v
        dp = [[0, 0] for _ in range(n)]
        parent = [-1] * n
        
        # DFS to establish parent-child relationships
        visited = [False] * n
        stack = [root]
        visited[root] = True
        order = []
        
        while stack:
            v = stack.pop()
            order.append(v)
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    parent[u] = v
                    stack.append(u)
        
        # Process in reverse order (leaves to root)
        for v in reversed(order):
            children = [u for u in adj[v] if parent[u] == v]
            
            if not children:
                # Leaf node
                dp[v][0] = 0
                dp[v][1] = 1
            else:
                # Not including v: can take any combination of children
                dp[v][0] = sum(max(dp[c][0], dp[c][1]) for c in children)
                # Including v: cannot include any children
                dp[v][1] = 1 + sum(dp[c][0] for c in children)
        
        # Reconstruct solution
        result = []
        
        def reconstruct(v, include):
            if include:
                result.append(v)
                for u in adj[v]:
                    if parent[u] == v:
                        reconstruct(u, False)
            else:
                for u in adj[v]:
                    if parent[u] == v:
                        # Choose the better option
                        if dp[u][1] > dp[u][0]:
                            reconstruct(u, True)
                        else:
                            reconstruct(u, False)
        
        if dp[root][1] >= dp[root][0]:
            reconstruct(root, True)
        else:
            reconstruct(root, False)
        
        return sorted(result)
    
    def _is_bipartite(self, problem, n):
        """Check if graph is bipartite using BFS coloring."""
        color = [-1] * n
        
        for start in range(n):
            if color[start] != -1:
                continue
            
            queue = [start]
            color[start] = 0
            
            while queue:
                v = queue.pop(0)
                for u in range(n):
                    if problem[v][u] == 1:
                        if color[u] == -1:
                            color[u] = 1 - color[v]
                            queue.append(u)
                        elif color[u] == color[v]:
                            return False, None
        
        return True, color
    
    def _solve_bipartite(self, problem, n, color):
        """Solve maximum independent set for bipartite graph using matching."""
        # Create NetworkX graph for maximum matching
        G = nx.Graph()
        G.add_nodes_from(range(n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    G.add_edge(i, j)
        
        # Maximum matching in bipartite graph
        matching = nx.max_weight_matching(G)
        
        # Minimum vertex cover from matching
        covered = set()
        for u, v in matching:
            covered.add(u)
            covered.add(v)
        
        # Maximum independent set = vertices not in minimum vertex cover
        return sorted([i for i in range(n) if i not in covered])
    
    def _solve_graph(self, problem, n):
        """Solve the graph, detecting special cases."""
        # Check if bipartite
        is_bip, color = self._is_bipartite(problem, n)
        if is_bip:
            return self._solve_bipartite(problem, n, color)
        
        # Fall back to CP-SAT for general graphs
        return self._solve_cpsat(problem, n)
    
    def _solve_cpsat(self, problem, n):
        """Solve using CP-SAT with optimized parameters."""
        model = cp_model.CpModel()
        
        # Create boolean variables
        nodes = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        
        # Add independence constraints (only for edges that exist)
        for i in range(n):
            for j in range(i + 1, n):
                if problem[i][j] == 1:
                    model.Add(nodes[i] + nodes[j] <= 1)
        
        # Objective: Maximize the number of vertices chosen
        model.Maximize(sum(nodes))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        
        # Use multiple workers for parallelism
        solver.parameters.num_search_workers = 4
        solver.parameters.log_search_progress = False
        
        # Optimization parameters
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2
        solver.parameters.max_time_in_seconds = 10.0
        
        # Enable symmetry detection
        solver.parameters.symmetry_level = 2
        
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return [i for i in range(n) if solver.Value(nodes[i]) == 1]
        else:
            return []