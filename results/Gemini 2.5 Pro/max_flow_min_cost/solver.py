from ortools.linear_solver import pywraplp
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> list[list[Any]]:
        """
        Solves the maximum flow min cost problem using a single, optimized LP formulation.
        Model construction is accelerated by pre-building adjacency lists to avoid
        inefficient nested loops, which is beneficial for sparse graphs.
        """
        capacity = problem["capacity"]
        cost = problem["cost"]
        s = problem["s"]
        t = problem["t"]
        n = len(capacity)

        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            return [[0.0] * n for _ in range(n)]

        # Define flow variables and build adjacency lists simultaneously.
        x = {}
        out_edges = [[] for _ in range(n)]
        in_edges = [[] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if capacity[i][j] > 1e-9:
                    x[i, j] = solver.NumVar(0, capacity[i][j], f'x_{i}_{j}')
                    out_edges[i].append(j)
                    in_edges[j].append(i)

        # Define flow conservation constraints using adjacency lists.
        for i in range(n):
            if i != s and i != t:
                constraint = solver.Constraint(0, 0)
                for j in out_edges[i]:
                    constraint.SetCoefficient(x[i, j], 1)
                for j in in_edges[i]:
                    constraint.SetCoefficient(x[j, i], -1)

        # Define composite objective: Maximize M * Flow - Cost.
        objective = solver.Objective()
        objective.SetMaximization()

        # Choose a large M to prioritize maximizing flow.
        max_abs_cost_sum = 1.0
        for i, j in x.keys():
             max_abs_cost_sum += abs(cost[i][j]) * capacity[i][j]
        M = max_abs_cost_sum

        # Set objective coefficients efficiently.
        # Flow part: M * (sum of flows out of s - sum of flows into s)
        for j in out_edges[s]:
            objective.SetCoefficient(x[s, j], M)
        for j in in_edges[s]:
            objective.SetCoefficient(x[j, s], -M)
        
        # Cost part: - sum of cost[i,j] * x[i,j]
        for (i, j), var in x.items():
            current_coeff = objective.GetCoefficient(var)
            objective.SetCoefficient(var, current_coeff - cost[i][j])

        status = solver.Solve()

        if status != pywraplp.Solver.OPTIMAL:
            return [[0.0] * n for _ in range(n)]

        solution = [[0.0] * n for _ in range(n)]
        for (i, j), var in x.items():
            flow_val = var.solution_value()
            solution[i][j] = max(0.0, flow_val)
        
        return solution