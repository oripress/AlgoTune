from typing import Any, List
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> List[List[int]]:
        """
        VRP solver using CP-SAT (MTZ subtour elimination). Returns K routes,
        each starting and ending at the depot. This implementation follows the
        reference approach to ensure optimality.
        """
        D = problem["D"]
        K = int(problem["K"])
        depot = int(problem["depot"])
        n = len(D)

        # List of non-depot nodes
        nodes = [i for i in range(n) if i != depot]
        m = len(nodes)

        # Sanity checks
        if n == 0:
            return []
        if K <= 0:
            return []
        # If there are no customer nodes, return K trivial depot-only routes
        if m == 0:
            return [[depot, depot] for _ in range(K)]
        # If more vehicles than customers, infeasible in this model
        if K > m:
            return []

        model = cp_model.CpModel()

        # Binary variables x[i,j] = 1 if arc i->j is used
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

        # MTZ subtour elimination variables u[i]
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
                # u[i] + 1 <= u[j] + (n-1)*(1 - x[i,j])
                model.Add(u[i] + 1 <= u[j] + (n - 1) * (1 - x[(i, j)]))

        # Objective: minimize total distance
        model.Minimize(sum(D[i][j] * x[(i, j)] for (i, j) in x))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            routes: List[List[int]] = []
            # Reconstruct routes by following arcs from depot
            for j in range(n):
                if j != depot and solver.Value(x[(depot, j)]) == 1:
                    route = [depot, j]
                    current = j
                    # follow until returning to depot
                    while current != depot:
                        for k in range(n):
                            if current != k and solver.Value(x[(current, k)]) == 1:
                                route.append(k)
                                current = k
                                break
                    routes.append(route)
            return routes
        else:
            return []