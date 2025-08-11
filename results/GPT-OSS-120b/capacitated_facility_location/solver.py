from typing import Any, Dict, List

import numpy as np
from ortools.sat.python import cp_model

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exact Capacitated Facility Location solver using OR‑Tools CP‑SAT.

        The model:
            - y_i : 1 if facility i is opened
            - x_ij: 1 if customer j is served by facility i
            - each customer assigned to exactly one facility
            - facility capacity constraints
            - linking constraints x_ij <= y_i
            - objective = sum_i f_i*y_i + sum_{i,j} c_ij*x_ij

        Returns a dictionary with:
            - objective_value: optimal total cost (float)
            - facility_status: list of booleans for each facility
            - assignments: list of lists (n_facilities × n_customers) with 0.0/1.0
        """
        # ----- Parse input -------------------------------------------------
        fixed_costs = np.array(problem["fixed_costs"], dtype=float)
        capacities = np.array(problem["capacities"], dtype=float)
        demands = np.array(problem["demands"], dtype=float)
        transport = np.array(problem["transportation_costs"], dtype=float)

        n_facilities = fixed_costs.size
        n_customers = demands.size

        # ----- Scale to integers (preserve 3 decimal places) ---------------
        scale = 1000  # keep three decimal digits
        int_fixed = (fixed_costs * scale).round().astype(int).tolist()
        int_cap = (capacities * scale).round().astype(int).tolist()
        int_dem = (demands * scale).round().astype(int).tolist()
        int_trans = (transport * scale).round().astype(int).tolist()

        # ----- Build CP‑SAT model -----------------------------------------
        model = cp_model.CpModel()

        # Decision variables
        y = [model.NewBoolVar(f"y_{i}") for i in range(n_facilities)]
        x = [
            [model.NewBoolVar(f"x_{i}_{j}") for j in range(n_customers)]
            for i in range(n_facilities)
        ]

        # Each customer assigned to exactly one facility
        for j in range(n_customers):
            model.Add(sum(x[i][j] for i in range(n_facilities)) == 1)

        # Capacity constraints
        for i in range(n_facilities):
            model.Add(
                sum(int_dem[j] * x[i][j] for j in range(n_customers))
                <= int_cap[i] * y[i]
            )

        # Linking constraints: a customer can be assigned only if facility is open
        for i in range(n_facilities):
            for j in range(n_customers):
                model.Add(x[i][j] <= y[i])

        # Objective: fixed costs + transportation costs
        obj_terms: List[cp_model.LinearExpr] = []
        for i in range(n_facilities):
            obj_terms.append(int_fixed[i] * y[i])
            for j in range(n_customers):
                obj_terms.append(int_trans[i][j] * x[i][j])
        model.Minimize(sum(obj_terms))

        # ----- Solve -------------------------------------------------------
        solver = cp_model.CpSolver()
        # Allow up to 30 seconds; most test instances solve much faster.
        solver.parameters.max_time_in_seconds = 30.0
        # Use all available threads
        solver.parameters.num_search_workers = max(1, cp_model.CpSolver().parameters.num_search_workers)

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Infeasible or unknown – return a placeholder indicating failure
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }

        # ----- Extract solution --------------------------------------------
        facility_status = [bool(solver.Value(y[i])) for i in range(n_facilities)]
        assignments = [
            [float(solver.Value(x[i][j])) for j in range(n_customers)]
            for i in range(n_facilities)
        ]

        # Convert objective back to original scale
        objective_value = solver.ObjectiveValue() / scale

        return {
            "objective_value": objective_value,
            "facility_status": facility_status,
            "assignments": assignments,
        }