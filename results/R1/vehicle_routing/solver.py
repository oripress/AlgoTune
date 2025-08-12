from ortools.sat.python import cp_model
from typing import Any
import time

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        model = cp_model.CpModel()

        # x[i,j] = 1 if arc i->j is used
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

        # Each non-depot node must be entered exactly once and left exactly once
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[(j, i)] for j in range(n) if j != i) == 1)
            model.Add(sum(x[(i, j)] for j in range(n) if j != i) == 1)

        # Depot must have exactly K departures and K arrivals
        model.Add(sum(x[(depot, j)] for j in range(n) if j != depot) == K)
        model.Add(sum(x[(i, depot)] for i in range(n) if i != depot) == K)

        # MTZ subtour elimination
        u = {}
        for i in range(n):
            if i == depot:
                continue
            u[i] = model.NewIntVar(1, n - 1, f"u_{i}")
        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or i == j:
                    continue
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[(i, j)]))

        # Objective: minimize total distance
        model.Minimize(sum(D[i][j] * x[(i, j)] for i, j in x))

        solver = cp_model.CpSolver()
        # Enable parallel solving with 8 workers
        solver.parameters.num_search_workers = 8
        # Set a 10-second time limit for each problem
        solver.parameters.max_time_in_seconds = 10.0
        
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            routes: list[list[int]] = []
            # Reconstruct routes by following arcs from depot
            for j in range(n):
                if j != depot and solver.Value(x[(depot, j)]) == 1:
                    route = [depot, j]
                    current = j
                    while current != depot:
                        for k in range(n):
                            if current != k and solver.Value(x[(current, k)]) == 1:
                                route.append(k)
                                current = k
                                break
                    routes.append(route)
            return routes
        else:
            # Return a fallback solution if no optimal found
            return [[depot] for _ in range(K)]
        routes = []
        if solution:
            for vehicle_id in range(K):
                index = routing.Start(vehicle_id)
                route = [manager.IndexToNode(index)]
                while not routing.IsEnd(index):
                    index = solution.Value(routing.NextVar(index))
                    node = manager.IndexToNode(index)
                    route.append(node)
                routes.append(route)
        return routes