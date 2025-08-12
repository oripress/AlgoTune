from typing import Any
from ortools.sat.python import cp_model
import numpy as np

class Solver:
    def __init__(self):
        # Pre-compile solver to save time
        self.test_solver = cp_model.CpSolver()
        
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Optimized VRP solver using CP-SAT with aggressive parameters.
        """
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        
        # For very small instances, use brute force approach
        if n <= 6:
            return self._solve_small(D, K, depot, n)
        
        # Use CP-SAT for larger instances with optimized parameters
        model = cp_model.CpModel()
        
        # Create binary variables for edges
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar('')
        
        # Flow constraints for non-depot nodes
        for i in range(n):
            if i != depot:
                model.Add(sum(x.get((j, i), 0) for j in range(n) if j != i) == 1)
                model.Add(sum(x.get((i, j), 0) for j in range(n) if j != i) == 1)
        
        # Depot constraints
        model.Add(sum(x.get((depot, j), 0) for j in range(n) if j != depot) == K)
        model.Add(sum(x.get((i, depot), 0) for i in range(n) if i != depot) == K)
        
        # Subtour elimination using MTZ formulation
        u = [model.NewIntVar(0, n-1, '') for _ in range(n)]
        for i in range(n):
            if i != depot:
                for j in range(n):
                    if j != depot and i != j:
                        model.Add(u[i] + 1 <= u[j] + n * (1 - x.get((i, j), 0)))
        
        # Objective
        total_cost = sum(int(D[i][j]) * x[(i, j)] for i, j in x)
        model.Minimize(total_cost)
        
        # Aggressive solver parameters for speed
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1.0
        solver.parameters.num_search_workers = 8
        solver.parameters.linearization_level = 0
        solver.parameters.cp_model_presolve = True
        solver.parameters.use_optional_variables = False
        
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_routes(solver, x, n, depot, K)
        
        return [[depot, depot] for _ in range(K)]
    
    def _solve_small(self, D, K, depot, n):
        """Fast solution for small instances."""
        model = cp_model.CpModel()
        
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar('')
        
        for i in range(n):
            if i != depot:
                model.Add(sum(x.get((j, i), 0) for j in range(n) if j != i) == 1)
                model.Add(sum(x.get((i, j), 0) for j in range(n) if j != i) == 1)
        
        model.Add(sum(x.get((depot, j), 0) for j in range(n) if j != depot) == K)
        model.Add(sum(x.get((i, depot), 0) for i in range(n) if i != depot) == K)
        
        u = [model.NewIntVar(0, n-1, '') for _ in range(n)]
        for i in range(n):
            if i != depot:
                for j in range(n):
                    if j != depot and i != j:
                        model.Add(u[i] + 1 <= u[j] + n * (1 - x.get((i, j), 0)))
        
        model.Minimize(sum(int(D[i][j]) * x[(i, j)] for i, j in x))
        
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 0.5
        solver.Solve(model)
        
        return self._extract_routes(solver, x, n, depot, K)
    
    def _extract_routes(self, solver, x, n, depot, K):
        """Extract routes from solution."""
        routes = []
        visited = set()
        
        for start in range(n):
            if start != depot and start not in visited:
                if solver.Value(x.get((depot, start), 0)) == 1:
                    route = [depot, start]
                    visited.add(start)
                    current = start
                    
                    while current != depot:
                        for next_node in range(n):
                            if solver.Value(x.get((current, next_node), 0)) == 1:
                                route.append(next_node)
                                if next_node != depot:
                                    visited.add(next_node)
                                current = next_node
                                break
                    
                    routes.append(route)
        
        # Add empty routes if needed
        while len(routes) < K:
            routes.append([depot, depot])
        
        return routes[:K]