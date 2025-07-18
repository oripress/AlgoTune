from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)
        model = cp_model.CpModel()
        # Create binary vars x[i,j]
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")
        # Each non-depot node enters and leaves exactly once
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[(j, i)] for j in range(n) if j != i) == 1)
            model.Add(sum(x[(i, j)] for j in range(n) if j != i) == 1)
        # Depot has K departures and arrivals
        model.Add(sum(x[(depot, j)] for j in range(n) if j != depot) == K)
        model.Add(sum(x[(i, depot)] for i in range(n) if i != depot) == K)
        # MTZ variables
        u = {}
        for i in range(n):
            if i == depot:
                continue
            u[i] = model.NewIntVar(1, n - 1, f"u_{i}")
        # MTZ constraints
        for i in range(n):
            if i == depot:
                continue
            for j in range(n):
                if j == depot or j == i:
                    continue
                model.Add(u[i] + 1 <= u[j] + (n - 1)*(1 - x[(i, j)]))
        # Objective: minimize total distance
        model.Minimize(sum(D[i][j] * x[(i, j)] for i, j in x))
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []
        # Reconstruct routes by following arcs from depot
        routes = []
        for j in range(n):
            if j != depot and solver.Value(x[(depot, j)]) == 1:
                route = [depot, j]
                current = j
                while current != depot:
                    for k in range(n):
                        if k != current and solver.Value(x[(current, k)]) == 1:
                            route.append(k)
                            current = k
                            break
                routes.append(route)
        return routes