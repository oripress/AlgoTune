from typing import Any
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem exactly using OR-Tools CBC solver.
        """
        # Problem data
        fixed_costs = problem["fixed_costs"]
        capacities = problem["capacities"]
        demands = problem["demands"]
        costs = problem["transportation_costs"]
        n_fac = len(fixed_costs)
        m_cust = len(demands)

        # Create MIP solver with CBC backend
        solver = pywraplp.Solver.CreateSolver('CBC')
        if solver is None:
            solver = pywraplp.Solver('CbcSolver', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

        # Decision variables
        y = [solver.IntVar(0, 1, f'y[{i}]') for i in range(n_fac)]
        x = [
            [solver.IntVar(0, 1, f'x[{i},{j}]') for j in range(m_cust)]
            for i in range(n_fac)
        ]

        # Constraints: each customer served by exactly one facility
        for j in range(m_cust):
            solver.Add(solver.Sum(x[i][j] for i in range(n_fac)) == 1)

        # Capacity and linkage constraints
        for i in range(n_fac):
            # capacity constraint
            solver.Add(solver.Sum(demands[j] * x[i][j] for j in range(m_cust)) <= capacities[i] * y[i])
            # assignment only if facility open
            for j in range(m_cust):
                solver.Add(x[i][j] <= y[i])

        # Objective: fixed costs + transportation costs
        objective = solver.Objective()
        for i in range(n_fac):
            objective.SetCoefficient(y[i], fixed_costs[i])
            for j in range(m_cust):
                objective.SetCoefficient(x[i][j], costs[i][j])
        objective.SetMinimization()

        # Optional time limit (milliseconds)
        if "time_limit" in kwargs:
            try:
                solver.set_time_limit(int(kwargs["time_limit"]))
            except Exception:
                pass

        # Solve
        status = solver.Solve()

        # Check feasible or optimal
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_fac,
                "assignments": [[0.0] * m_cust for _ in range(n_fac)],
            }

        # Extract solution
        facility_status = [bool(y[i].solution_value()) for i in range(n_fac)]
        assignments = [
            [float(x[i][j].solution_value()) for j in range(m_cust)]
            for i in range(n_fac)
        ]
        objective_value = solver.Objective().Value()

        return {
            "objective_value": float(objective_value),
            "facility_status": facility_status,
            "assignments": assignments,
        }