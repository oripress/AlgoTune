from typing import Any
from ortools.linear_solver import pywraplp
import numpy as np

class Solver:
    def solve(self, problem, **kwargs) -> Any:
        fixed_costs = problem["fixed_costs"]
        capacities = problem["capacities"]
        demands = problem["demands"]
        transportation_costs = problem["transportation_costs"]
        
        n_facilities = len(fixed_costs)
        n_customers = len(demands)
        
        # Create solver using HIGHS
        solver = pywraplp.Solver.CreateSolver('HIGHS')
        if not solver:
            solver = pywraplp.Solver.CreateSolver('CBC')
        
        # Set solver parameters for speed
        solver.SetNumThreads(4)
        
        # Variables - use NumVar with bounds instead of IntVar for relaxation + branch
        y = [solver.IntVar(0, 1, '') for i in range(n_facilities)]
        
        # Flatten x variables for faster access
        x_flat = [solver.IntVar(0, 1, '') for _ in range(n_facilities * n_customers)]
        
        # Objective using direct coefficient setting (faster)
        objective = solver.Objective()
        for i in range(n_facilities):
            objective.SetCoefficient(y[i], fixed_costs[i])
        
        for i in range(n_facilities):
            for j in range(n_customers):
                objective.SetCoefficient(x_flat[i * n_customers + j], transportation_costs[i][j])
        objective.SetMinimization()
        
        # Constraints: each customer must be served by exactly one facility
        for j in range(n_customers):
            ct = solver.Constraint(1, 1)
            for i in range(n_facilities):
                ct.SetCoefficient(x_flat[i * n_customers + j], 1)
        
        # Capacity constraints: sum_j d_j * x_ij <= s_i * y_i
        for i in range(n_facilities):
            ct = solver.Constraint(-solver.infinity(), 0)
            base = i * n_customers
            for j in range(n_customers):
                ct.SetCoefficient(x_flat[base + j], demands[j])
            ct.SetCoefficient(y[i], -capacities[i])
        
        # x[i][j] <= y[i] constraints
        for i in range(n_facilities):
            base = i * n_customers
            for j in range(n_customers):
                ct = solver.Constraint(-solver.infinity(), 0)
                ct.SetCoefficient(x_flat[base + j], 1)
                ct.SetCoefficient(y[i], -1)
        
        # Solve
        status = solver.Solve()
        
        if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            facility_status = [y[i].solution_value() > 0.5 for i in range(n_facilities)]
            assignments = [[x_flat[i * n_customers + j].solution_value() 
                           for j in range(n_customers)] for i in range(n_facilities)]
            return {
                "objective_value": solver.Objective().Value(),
                "facility_status": facility_status,
                "assignments": assignments,
            }
        else:
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }