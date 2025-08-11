from typing import Any, Dict, List

import numpy as np
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # Parse input
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=float)
        capacities = np.asarray(problem["capacities"], dtype=float)
        demands = np.asarray(problem["demands"], dtype=float)
        transportation_costs = np.asarray(problem["transportation_costs"], dtype=float)

        n_facilities = fixed_costs.size
        n_customers = demands.size

        # Validate shapes
        if transportation_costs.shape != (n_facilities, n_customers):
            # Invalid input shape; return infeasible
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Quick infeasibility checks
        total_demand = float(demands.sum())
        if float(capacities.sum()) + 1e-12 < total_demand:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Build solver
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            # Fallback infeasible (CBC should be available)
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Decision variables
        # y_i in {0,1}
        y_vars: List[pywraplp.Variable] = [solver.BoolVar(f"y_{i}") for i in range(n_facilities)]

        # x_ij in {0,1}; if a single customer's demand exceeds capacity of facility, fix x_ij = 0
        x_vars: List[List[pywraplp.Variable]] = []
        for i in range(n_facilities):
            row = []
            cap_i = capacities[i]
            for j in range(n_customers):
                if demands[j] > cap_i + 1e-12:
                    # Impossible assignment due to capacity; fix to 0
                    var = solver.IntVar(0.0, 0.0, f"x_{i}_{j}")
                else:
                    var = solver.BoolVar(f"x_{i}_{j}")
                row.append(var)
            x_vars.append(row)

        # Constraints
        # Each customer assigned exactly once
        for j in range(n_customers):
            ct = solver.RowConstraint(1.0, 1.0, f"assign_{j}")
            for i in range(n_facilities):
                ct.SetCoefficient(x_vars[i][j], 1.0)

        # Capacity and linking constraints
        for i in range(n_facilities):
            # Sum_j d_j x_ij - s_i y_i <= 0
            cap_ct = solver.RowConstraint(-solver.infinity(), 0.0, f"cap_{i}")
            cap_ct.SetCoefficient(y_vars[i], -capacities[i])
            for j in range(n_customers):
                d = demands[j]
                if d != 0.0:
                    cap_ct.SetCoefficient(x_vars[i][j], d)

            # Link: x_ij - y_i <= 0  => x_ij <= y_i
            for j in range(n_customers):
                link_ct = solver.RowConstraint(-solver.infinity(), 0.0, f"link_{i}_{j}")
                link_ct.SetCoefficient(x_vars[i][j], 1.0)
                link_ct.SetCoefficient(y_vars[i], -1.0)

        # Objective: sum_i f_i y_i + sum_{i,j} c_{ij} x_{ij}
        objective = solver.Objective()
        for i in range(n_facilities):
            coeff = float(fixed_costs[i])
            if coeff != 0.0:
                objective.SetCoefficient(y_vars[i], coeff)
        for i in range(n_facilities):
            costs_i = transportation_costs[i]
            for j in range(n_customers):
                c = float(costs_i[j])
                if c != 0.0:
                    objective.SetCoefficient(x_vars[i][j], c)
        objective.SetMinimization()

        # Solve
        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # Extract solution
        facility_status = [bool(y_vars[i].solution_value() > 0.5) for i in range(n_facilities)]
        assignments: List[List[float]] = []
        for i in range(n_facilities):
            row = []
            for j in range(n_customers):
                val = 1.0 if x_vars[i][j].solution_value() > 0.5 else 0.0
                row.append(val)
            assignments.append(row)

        # Compute objective explicitly from integral solution
        obj_val = float(np.dot(fixed_costs, np.array(facility_status, dtype=float)))
        obj_val += float(np.sum(transportation_costs * np.array(assignments, dtype=float)))

        return {
            "objective_value": obj_val,
            "facility_status": facility_status,
            "assignments": assignments,
        }