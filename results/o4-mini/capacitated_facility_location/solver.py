#!/usr/bin/env python3
from typing import Any
from ortools.sat.python import cp_model
import os

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem using OR-Tools CP-SAT solver.
        """
        # Extract problem data
        fixed_costs = problem.get("fixed_costs", [])
        capacities = problem.get("capacities", [])
        demands = problem.get("demands", [])
        transport = problem.get("transportation_costs", [])
        nF = len(fixed_costs)
        nC = len(demands)

        # Scale floats to large-int domain for CP-SAT
        scale = 10**9
        fixed_i = [int(round(f * scale)) for f in fixed_costs]
        cap_i = [int(round(s * scale)) for s in capacities]
        dem_j = [int(round(d * scale)) for d in demands]
        cost_ij = [[int(round(c * scale)) for c in row] for row in transport]

        # Build model
        model = cp_model.CpModel()
        # Decision vars
        y = [model.NewBoolVar(f'y[{i}]') for i in range(nF)]
        x = [[model.NewBoolVar(f'x[{i},{j}]') for j in range(nC)] for i in range(nF)]

        # Each customer assigned exactly once
        for j in range(nC):
            model.Add(sum(x[i][j] for i in range(nF)) == 1)

        # Capacity linking: sum demand*assign <= capacity * open
        for i in range(nF):
            model.Add(sum(dem_j[j] * x[i][j] for j in range(nC)) <= cap_i[i] * y[i])

        # Objective: open costs + transport costs
        model.Minimize(
            sum(fixed_i[i] * y[i] for i in range(nF)) +
            sum(cost_ij[i][j] * x[i][j] for i in range(nF) for j in range(nC))
        )

        # Solve with all CPUs
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = os.cpu_count() or 1
        status = solver.Solve(model)

        # Infeasible?
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * nF,
                "assignments": [[0.0] * nC for _ in range(nF)],
            }

        # Retrieve booleans
        facility_status = [bool(solver.Value(y[i])) for i in range(nF)]
        assignments = [
            [float(solver.Value(x[i][j])) for j in range(nC)] for i in range(nF)
        ]

        # Compute real objective in floats
        obj = 0.0
        for i in range(nF):
            if facility_status[i]:
                obj += fixed_costs[i]
            for j in range(nC):
                if assignments[i][j] > 0.5:
                    obj += transport[i][j]

        return {
            "objective_value": obj,
            "facility_status": facility_status,
            "assignments": assignments,
        }