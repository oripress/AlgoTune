from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Solve the VRP problem using CP-SAT solver.

        :param problem: Dict with "D", "K", and "depot".
        :return: A list of K routes, each a list of nodes starting and ending at the depot.
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
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
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 1
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL:
            routes: list[list[int]] = []
            
            # Build adjacency list for O(1) lookups instead of O(n) scanning
            next_node = {}
            for i in range(n):
                for j in range(n):
                    if i != j and solver.Value(x[(i, j)]) == 1:
                        next_node[i] = j
            
            # Reconstruct routes efficiently
            for j in range(n):
                if j != depot and solver.Value(x[(depot, j)]) == 1:
                    route = [depot, j]
                    current = j
                    while current != depot:
                        current = next_node[current]
                        route.append(current)
                    routes.append(route)
            
            return routes
        else:
            return []