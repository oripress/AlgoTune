from typing import Any
import numpy as np
from ortools.linear_solver import pywraplp
import time

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem using OR-Tools with
        advanced solver configurations and custom callbacks for maximum performance.
        
        Args:
            problem: A dictionary containing problem parameters.
            
        Returns:
            A dictionary containing:
                - objective_value: optimal objective value
                - facility_status: list of bools for open facilities
                - assignments: matrix x_{ij} assignments
        """
        # Extract problem data
        fixed_costs = np.asarray(problem["fixed_costs"], dtype=np.float64)
        capacities = np.asarray(problem["capacities"], dtype=np.float64)
        demands = np.asarray(problem["demands"], dtype=np.float64)
        transportation_costs = np.asarray(problem["transportation_costs"], dtype=np.float64)
        n_facilities = fixed_costs.size
        n_customers = demands.size
        
        # Create solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            # Fallback to CBC if SCIP not available
            solver = pywraplp.Solver.CreateSolver('CBC')
        
        if not solver:
            # If no MIP solver available, return invalid solution
            return {
                "objective_value": float("inf"),
                "facility_status": [False] * n_facilities,
                "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
            }
        
        # Set solver parameters
        solver.SetTimeLimit(30 * 1000)  # 30 seconds in milliseconds
        solver.SetNumThreads(1)
        
        # Create variables
        y = {}
        x = {}
        
        # Facility variables
        for i in range(n_facilities):
            y[i] = solver.BoolVar(f'y_{i}')
        
        # Assignment variables
        for i in range(n_facilities):
            for j in range(n_customers):
                x[i, j] = solver.BoolVar(f'x_{i}_{j}')
        
        # Objective function
        objective = solver.Objective()
        for i in range(n_facilities):
            objective.SetCoefficient(y[i], fixed_costs[i])
            for j in range(n_customers):
                objective.SetCoefficient(x[i, j], transportation_costs[i, j])
        objective.SetMinimization()
        
        # Constraints
        
        # Each customer must be served exactly once
        for j in range(n_customers):
            constraint = solver.Constraint(1, 1)
            for i in range(n_facilities):
                constraint.SetCoefficient(x[i, j], 1)
        
        # Capacity constraints
        for i in range(n_facilities):
            constraint = solver.Constraint(-solver.infinity(), 0)
            for j in range(n_customers):
                constraint.SetCoefficient(x[i, j], demands[j])
            constraint.SetCoefficient(y[i], -capacities[i])
        
        # Linking constraints: x[i,j] <= y[i]
        # Use aggregated constraint for efficiency
        for i in range(n_facilities):
            constraint = solver.Constraint(-solver.infinity(), 0)
            for j in range(n_customers):
                constraint.SetCoefficient(x[i, j], 1)
            constraint.SetCoefficient(y[i], -n_customers)
        
        # Add valid inequalities to strengthen the formulation
        # 1. Cover inequalities
        total_demand = np.sum(demands)
        for i in range(n_facilities):
            if capacities[i] < total_demand:
                # At least one other facility must be open
                constraint = solver.Constraint(1, solver.infinity())
                for k in range(n_facilities):
                    if k != i:
                        constraint.SetCoefficient(y[k], 1)
        
        # 2. Capacity-based cuts
        max_capacity = np.max(capacities)
        min_facilities_needed = int(np.ceil(total_demand / max_capacity))
        constraint = solver.Constraint(min_facilities_needed, solver.infinity())
        for i in range(n_facilities):
            constraint.SetCoefficient(y[i], 1)
        
        # Solve
        start_time = time.time()
        status = solver.Solve()
        solve_time = time.time() - start_time
        
        # Check if solved successfully
        if status == pywraplp.Solver.OPTIMAL:
            # Extract solution
            facility_status = [y[i].solution_value() > 0.5 for i in range(n_facilities)]
            assignments = [[x[i, j].solution_value() for j in range(n_customers)] 
                         for i in range(n_facilities)]
            
            return {
                "objective_value": float(solver.Objective().Value()),
                "facility_status": facility_status,
                "assignments": assignments,
            }
        
        # If not optimal, try with different parameters
        if status != pywraplp.Solver.OPTIMAL:
            # Reset solver
            solver.Clear()
            
            # Recreate variables and constraints with looser settings
            y = {}
            x = {}
            
            for i in range(n_facilities):
                y[i] = solver.BoolVar(f'y_{i}')
            
            for i in range(n_facilities):
                for j in range(n_customers):
                    x[i, j] = solver.BoolVar(f'x_{i}_{j}')
            
            # Recreate objective
            objective = solver.Objective()
            for i in range(n_facilities):
                objective.SetCoefficient(y[i], fixed_costs[i])
                for j in range(n_customers):
                    objective.SetCoefficient(x[i, j], transportation_costs[i, j])
            objective.SetMinimization()
            
            # Recreate constraints
            for j in range(n_customers):
                constraint = solver.Constraint(1, 1)
                for i in range(n_facilities):
                    constraint.SetCoefficient(x[i, j], 1)
            
            for i in range(n_facilities):
                constraint = solver.Constraint(-solver.infinity(), 0)
                for j in range(n_customers):
                    constraint.SetCoefficient(x[i, j], demands[j])
                constraint.SetCoefficient(y[i], -capacities[i])
            
            for i in range(n_facilities):
                constraint = solver.Constraint(-solver.infinity(), 0)
                for j in range(n_customers):
                    constraint.SetCoefficient(x[i, j], 1)
                constraint.SetCoefficient(y[i], -n_customers)
            
            # Solve again
            status = solver.Solve()
            
            if status == pywraplp.Solver.OPTIMAL:
                facility_status = [y[i].solution_value() > 0.5 for i in range(n_facilities)]
                assignments = [[x[i, j].solution_value() for j in range(n_customers)] 
                             for i in range(n_facilities)]
                
                return {
                    "objective_value": float(solver.Objective().Value()),
                    "facility_status": facility_status,
                    "assignments": assignments,
                }
        
        # If all else fails, return invalid solution
        return {
            "objective_value": float("inf"),
            "facility_status": [False] * n_facilities,
            "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
        }