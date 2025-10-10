from ortools.sat.python import cp_model
import numpy as np

class Solver:
    def solve(self, problem, **kwargs):
        """
        Optimized TSP solver using exact methods.
        """
        n = len(problem)
        
        if n <= 1:
            return [0, 0]
        
        # Use DP for small instances (much faster than OR-Tools)
        if n <= 18:
            return self.solve_dp(problem)
        else:
            # Use optimized OR-Tools for larger instances
            return self.solve_ortools(problem)
    
    def solve_dp(self, problem):
        """Held-Karp dynamic programming algorithm with numpy."""
        n = len(problem)
        
        # dp[mask][i] = minimum cost to visit cities in mask ending at i
        INF = float('inf')
        dp = [[INF] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        # Start at city 0
        dp[1][0] = 0
        
        # Iterate through all subsets
        for mask in range(1, 1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == INF:
                    continue
                
                curr_cost = dp[mask][last]
                
                # Try extending to next city
                for next_city in range(n):
                    if mask & (1 << next_city):
                        continue
                    
                    new_mask = mask | (1 << next_city)
                    new_cost = curr_cost + problem[last][next_city]
                    
                    if new_cost < dp[new_mask][next_city]:
                        dp[new_mask][next_city] = new_cost
                        parent[new_mask][next_city] = last
        
        # Find best ending city
        full_mask = (1 << n) - 1
        min_cost = INF
        best_last = -1
        
        for last in range(1, n):
            cost = dp[full_mask][last] + problem[last][0]
            if cost < min_cost:
                min_cost = cost
                best_last = last
        
        # Reconstruct path
        path = []
        mask = full_mask
        current = best_last
        
        while current != -1:
            path.append(current)
            prev = parent[mask][current]
            if prev == -1:
                break
            mask ^= (1 << current)
            current = prev
        
        path.reverse()
        path.insert(0, 0)
        path.append(0)
        
        return path
        return path
    
    def solve_ortools(self, problem):
        """Use OR-Tools CP-SAT solver with optimized parameters."""
        n = len(problem)
        model = cp_model.CpModel()
        
        # Create variables
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = model.NewBoolVar(f"x_{i}_{j}")
        
        # Circuit constraint
        arcs = []
        for (i, j), var in x.items():
            arcs.append((i, j, var))
        model.AddCircuit(arcs)
        
        # Minimize total distance
        model.Minimize(sum(problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
        solver.parameters.num_search_workers = 8
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            path = [0]
            current_city = 0
            while len(path) < n:
                for next_city in range(n):
                    if current_city != next_city and solver.Value(x[current_city, next_city]) == 1:
                        path.append(next_city)
                        current_city = next_city
                        break
            path.append(0)
            return path
        else:
            return []