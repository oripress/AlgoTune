class Solver:
    def solve(self, problem, **kwargs):
        """Solve the Bottleneck TSP problem."""
        n = len(problem)
        if n <= 1:
            return [0, 0]
        
        # For small instances, use exact method
        if n <= 10:
            return self._solve_exact(problem)
        
        # For larger instances, use binary search with Hamiltonian cycle detection
        return self._solve_binary_search(problem)
    
    def _solve_exact(self, problem):
        """Exact solution for small instances using permutations."""
        import itertools
        n = len(problem)
        
        best_tour = None
        best_bottleneck = float('inf')
        
        # Try all permutations starting from city 0
        for perm in itertools.permutations(range(1, n)):
            tour = [0] + list(perm) + [0]
            bottleneck = max(problem[tour[i]][tour[i+1]] for i in range(n))
            if bottleneck < best_bottleneck:
                best_bottleneck = bottleneck
                best_tour = tour
        
        return best_tour
    
    def _solve_binary_search(self, problem):
        """Use binary search on edge weights to find optimal bottleneck."""
        n = len(problem)
        
        # Collect all unique edge weights
        edge_weights = set()
        for i in range(n):
            for j in range(i + 1, n):
                edge_weights.add(problem[i][j])
        
        sorted_weights = sorted(edge_weights)
        
        # Binary search on the bottleneck value
        left, right = 0, len(sorted_weights) - 1
        best_tour = None
        
        while left <= right:
            mid = (left + right) // 2
            threshold = sorted_weights[mid]
            
            # Check if a Hamiltonian cycle exists with bottleneck <= threshold
            tour = self._find_hamiltonian_cycle(problem, n, threshold)
            
            if tour is not None:
                best_tour = tour
                right = mid - 1  # Try to find a smaller bottleneck
            else:
                left = mid + 1
        
        return best_tour if best_tour else self._fallback_solution(problem)
    
    def _find_hamiltonian_cycle(self, problem, n, threshold):
        """Find a Hamiltonian cycle using only edges with weight <= threshold."""
        # Build adjacency list with edges <= threshold
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and problem[i][j] <= threshold:
                    adj[i].append(j)
        
        # Check if degree constraint is satisfied (each node needs degree >= 2)
        for i in range(n):
            if len(adj[i]) < 2:
                return None
        
        # Try to find a Hamiltonian cycle starting from city 0
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        if self._dfs_hamiltonian(adj, path, visited, n, problem, threshold):
            path.append(0)
            return path
        
        return None
    
    def _dfs_hamiltonian(self, adj, path, visited, n, problem, threshold):
        """DFS to find Hamiltonian cycle."""
        if len(path) == n:
            # Check if we can return to start
            return problem[path[-1]][0] <= threshold
        
        current = path[-1]
        for next_city in adj[current]:
            if not visited[next_city]:
                visited[next_city] = True
                path.append(next_city)
                
                if self._dfs_hamiltonian(adj, path, visited, n, problem, threshold):
                    return True
                
                path.pop()
                visited[next_city] = False
        
        return False
    
    def _fallback_solution(self, problem):
        """Fallback nearest neighbor solution."""
        n = len(problem)
        unvisited = set(range(1, n))
        tour = [0]
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda city: problem[current][city])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        tour.append(0)
        return tour