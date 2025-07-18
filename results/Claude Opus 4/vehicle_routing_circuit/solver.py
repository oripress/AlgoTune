from typing import Any
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> list[list[int]]:
        """
        Solve the VRP using OR-Tools CP-SAT solver with optimizations.
        """
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
        # Add time limit for faster solving
        solver.parameters.max_time_in_seconds = 1.0
        status = solver.Solve(model)
        
        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return []
        
        # Reconstruct routes
        routes: list[list[int]] = []
        for i in range(K):
            route = [depot]
            current = depot
            while True:
                next_city = None
                for v in range(n):
                    if current != v and solver.Value(x[i].get((current, v), 0)) == 1:
                        next_city = v
                        break
                # if no outgoing arc or we've returned to depot, close the tour
                if next_city is None or next_city == depot:
                    route.append(depot)
                    break
                route.append(next_city)
                current = next_city
            routes.append(route)
        
        return routes