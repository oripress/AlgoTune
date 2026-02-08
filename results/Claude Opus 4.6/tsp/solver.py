import numpy as np
from typing import Any

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        n = len(problem)
        if n <= 1:
            return [0, 0]
        if n == 2:
            return [0, 1, 0]
        
        dist = problem
        
        # For small n, use Held-Karp DP (bitmask)
        if n <= 20:
            return self._held_karp(dist, n)
        else:
            return self._solve_large(dist, n)
    
    def _held_karp(self, dist, n):
        INF = float('inf')
        size = 1 << n
        
        dp = np.full((size, n), INF, dtype=np.float64)
        parent = np.full((size, n), -1, dtype=np.int32)
        
        dp[1][0] = 0
        
        for S in range(size):
            for u in range(n):
                if dp[S][u] == INF:
                    continue
                if not (S & (1 << u)):
                    continue
                for v in range(n):
                    if S & (1 << v):
                        continue
                    new_S = S | (1 << v)
                    new_cost = dp[S][u] + dist[u][v]
                    if new_cost < dp[new_S][v]:
                        dp[new_S][v] = new_cost
                        parent[new_S][v] = u
        
        full_mask = size - 1
        best_cost = INF
        best_last = -1
        for u in range(1, n):
            cost = dp[full_mask][u] + dist[u][0]
            if cost < best_cost:
                best_cost = cost
                best_last = u
        
        path = []
        S = full_mask
        u = best_last
        while u != -1:
            path.append(u)
            prev = parent[S][u]
            S = S ^ (1 << u)
            u = prev
        
        path.reverse()
        path.append(0)
        return path
    
    def _solve_large(self, dist, n):
        from ortools.sat.python import cp_model
        
        model = cp_model.CpModel()
        x = {(i, j): model.NewBoolVar(f"x[{i},{j}]") for i in range(n) for j in range(n) if i != j}
        model.AddCircuit([(u, v, var) for (u, v), var in x.items()])
        model.Minimize(sum(dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j))
        
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = False
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
            path.append(0)
            return path
        return []