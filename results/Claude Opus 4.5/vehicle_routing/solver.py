from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem, **kwargs) -> Any:
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

        # Each non-depot node: in-degree = out-degree = 1
        for i in range(n):
            if i == depot:
                continue
            model.Add(sum(x[(j, i)] for j in range(n) if j != i) == 1)
            model.Add(sum(x[(i, j)] for j in range(n) if j != i) == 1)

        # Depot: out-degree = in-degree = K
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

        # Objective: minimize total distance (scale to int)
        model.Minimize(sum(int(D[i][j] * 1000) * x[(i, j)] for i, j in x))

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            routes = []
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
        return []