import logging
import random
from typing import Any

from ortools.sat.python import cp_model

from AlgoTuneTasks.base import register_task, Task


@register_task("vehicle_routing_circuit")
class VehicleRouting(Task):
    def __init__(self, **kwargs):
        """
        Initialize the Vehicle Routing Task.

        :param kwargs: Keyword arguments.
        """
        super().__init__(**kwargs)

    def generate_problem(self, n: int, random_seed: int = 1) -> dict[str, Any]:
        """
        Generate a VRP instance with 5*n locations.

        :param n: Base scale; total number of locations (including depot) will be 2*n.
        :param random_seed: Seed for reproducibility.
        :raises ValueError: If n < 1 (need at least one vehicle).
        :return: A dict with keys:
                 - "D":      (2n x 2n) distance matrix (symmetric, zeros on diagonal)
                 - "K":      number of vehicles (set to n)
                 - "depot":  index of the depot (0)
        """
        n = max(n, 2)
        logging.debug(f"Starting generate_problem with n={n}, random_seed={random_seed}")

        random.seed(random_seed)
        total_nodes = 2 * n
        D = [[0] * total_nodes for _ in range(total_nodes)]
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i == j:
                    D[i][j] = 0
                else:
                    D[i][j] = random.randint(1, 100)
                    logging.debug(f"D[{i}][{j}] = {D[i][j]}")

        K = random.randint(2, max(2, n))
        depot = 0
        logging.debug(f"Generated VRP with {total_nodes} locations, K={K}, depot={depot}")
        return {"D": D, "K": K, "depot": depot}

    def solve(self, problem: dict[str, Any]) -> list[list[int]]:
        """
        Solve the CVRP using AddCircuit (multi-TSP style), requiring optimality
        and infinite solve time.
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
        # No time limit; require OPTIMAL solution
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            logging.error("No optimal solution found.")
            return []

        # Reconstruct routes (fixed: append depot only once at end)
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

    def is_solution(self, problem: dict[str, Any], solution: list[list[int]]) -> bool:
        """
        Check if the proposed solution is valid and optimal.

        Validity:
          1) Exactly K routes.
          2) Each route starts and ends at depot.
          3) Each non-depot node appears exactly once across all routes.
          4) All distances on routes are positive.

        Optimality:
          5) Total distance equals the optimal distance from self.solve().

        :param problem: Dict with "D", "K", and "depot".
        :param solution: List of routes to verify.
        :return: True if valid and optimal; False otherwise.
        """
        D = problem["D"]
        K = problem["K"]
        depot = problem["depot"]
        n = len(D)

        # Check number of routes
        if len(solution) != K:
            return False

        visited = set()
        total_dist = 0
        for route in solution:
            if len(route) < 2 or route[0] != depot or route[-1] != depot:
                return False
            for idx in range(len(route) - 1):
                u, v = route[idx], route[idx + 1]
                if not (0 <= u < n and 0 <= v < n):
                    return False
                dist = D[u][v]
                if dist <= 0:
                    return False
                total_dist += dist
            for node in route[1:-1]:
                if node == depot or node in visited:
                    return False
                visited.add(node)

        # Check all non-depot nodes are visited
        if visited != set(range(n)) - {depot}:
            return False

        # Check optimality
        optimal_routes = self.solve(problem)
        opt_dist = 0
        for route in optimal_routes:
            for idx in range(len(route) - 1):
                opt_dist += D[route[idx]][route[idx + 1]]

        return abs(total_dist - opt_dist) < 1e-6
