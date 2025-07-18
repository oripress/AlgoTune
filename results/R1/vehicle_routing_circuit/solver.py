import random
from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        model = cp_model.CpModel()

        # Binary vars: x[i][u,v] = 1 if vehicle i travels u->v
        x = [
            {
                (u, v): model.NewBoolVar(f"x_{i}_{u}_{v}")
                for u in range(n)
                for v in range(n)
                if u != v
            }
            for i in range(K)
        ]
        # visit[i][u] = 1 if vehicle i visits node u
        visit = [[model.NewBoolVar(f"visit_{i}_{u}") for u in range(n)] for i in range(K)]

        # Build circuit constraint per vehicle
        for i in range(K):
            arcs = []
            # real travel arcs
            for (u, v), var in x[i].items():
                arcs.append((u, v, var))
            # skip arcs to allow skipping nodes
            for u in range(n):
                arcs.append((u, u, visit[i][u].Not()))
            model.AddCircuit(arcs)

        # Depot must be visited by all vehicles; others exactly once total
        for u in range(n):
            if u == depot:
                for i in range(K):
                    model.Add(visit[i][u] == 1)
            else:
                model.Add(sum(visit[i][u] for i in range(K)) == 1)

        # Link x → visit: if x[i][u,v]==1 then visit[i][u] and visit[i][v] are 1
        for i in range(K):
            for (u, v), var in x[i].items():
                model.Add(var <= visit[i][u])
                model.Add(var <= visit[i][v])

        # Objective: minimize total distance
        model.Minimize(sum(D[u][v] * x[i][(u, v)] for i in range(K) for (u, v) in x[i]))

        solver = cp_model.CpSolver()
        # Set performance parameters
        solver.parameters.num_search_workers = 8  # Use multiple cores
        solver.parameters.max_time_in_seconds = 30.0  # Timeout after 30 seconds

        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            return []

        # Reconstruct routes using a next_map for each vehicle to avoid O(n^2) per vehicle
        routes: list[list[int]] = []
        for i in range(K):
            # Build a mapping from current node to next node for this vehicle
            next_map = {}
            for (u, v), var in x[i].items():
                if solver.Value(var) == 1:
                    next_map[u] = v

            route = [depot]
            current = depot

            # If the vehicle is used (i.e., there's an outgoing arc from depot)
            if depot in next_map:
                current = next_map[depot]
                while current != depot:
                    route.append(current)
                    current = next_map[current]
                route.append(depot)
            else:
                # Unused vehicle: just [depot, depot]
                route.append(depot)

            routes.append(route)

        return routes