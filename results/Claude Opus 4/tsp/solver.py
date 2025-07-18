import numpy as np
from typing import Any, List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: List[List[int]], **kwargs) -> Any:
        """
        Solve the TSP problem using a hybrid approach.
        
        :param problem: Distance matrix as a list of lists.
        :return: A list representing the optimal tour, starting and ending at city 0.
        """
        n = len(problem)
        
        if n <= 1:
            return [0, 0]
        
        if n == 2:
            return [0, 1, 0]
        
        # For small instances, use dynamic programming
        if n <= 20:
            return self._solve_dp(problem)
        else:
            # For larger instances, use OR-Tools with optimized parameters
            return self._solve_ortools(problem)
    
    def _solve_dp(self, problem: List[List[int]]) -> List[int]:
        """Dynamic programming solution for small instances."""
        n = len(problem)
        dist = np.array(problem, dtype=np.int32)
        
        # dp[mask][i] = minimum cost to visit all cities in mask ending at city i
        INF = float('inf')
        dp = np.full((1 << n, n), INF)
        parent = np.full((1 << n, n), -1, dtype=np.int32)
        
        # Base case: start from city 0
        dp[1][0] = 0
        
        # Fill DP table
        for mask in range(1, 1 << n):
            if not (mask & 1):  # Must include city 0
                continue
                
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                    
                for v in range(n):
                    if u == v or (mask & (1 << v)):
                        continue
                    
                    new_mask = mask | (1 << v)
                    new_cost = dp[mask][u] + dist[u][v]
                    
                    if new_cost < dp[new_mask][v]:
                        dp[new_mask][v] = new_cost
                        parent[new_mask][v] = u
        
        # Find minimum cost to visit all cities and return to 0
        all_mask = (1 << n) - 1
        min_cost = INF
        last_city = -1
        
        for i in range(1, n):
            cost = dp[all_mask][i] + dist[i][0]
            if cost < min_cost:
                min_cost = cost
                last_city = i
        
        # Reconstruct path
        path = []
        mask = all_mask
        curr = last_city
        
        while curr != -1:
            path.append(curr)
            prev = parent[mask][curr]
            if prev != -1:
                mask ^= (1 << curr)
            curr = prev
        
        path.reverse()
        path.append(0)  # Return to starting city
        
        return path
    
    def _solve_ortools(self, problem: List[List[int]]) -> List[int]:
        """OR-Tools solution for larger instances with optimized parameters."""
        n = len(problem)
        model = cp_model.CpModel()
        
        # Create variables
        x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in range(n) for j in range(n) if i != j}
        
        # Circuit constraint
        model.AddCircuit([(u, v, var) for (u, v), var in x.items()])
        
        # Add objective
        model.Minimize(sum(problem[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))
        
        # Solve with optimized parameters
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5.0  # Limit time
        solver.parameters.num_search_workers = 4  # Use multiple threads
        solver.parameters.log_search_progress = False  # Disable logging for speed
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            path = []
            current_city = 0
            while len(path) < n:
                path.append(current_city)
                for next_city in range(n):
                    if current_city != next_city and solver.Value(x[current_city, next_city]) == 1:
                        current_city = next_city
                        break
            path.append(0)  # Return to the starting city
            return path
        else:
            return []