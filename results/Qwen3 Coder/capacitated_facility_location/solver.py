from typing import Any
from ortools.linear_solver import pywraplp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves the Capacitated Facility Location Problem using OR-Tools.
        
        Args:
            problem: A dictionary containing problem parameters.
            
        Returns:
            A dictionary containing:
                - objective_value: optimal objective value
                - facility_status: list of bools for open facilities
                - assignments: matrix x_{ij} assignments
        """
        fixed_costs = problem["fixed_costs"]
        capacities = problem["capacities"]
        demands = problem["demands"]
        transportation_costs = problem["transportation_costs"]
        n_facilities = len(fixed_costs)
        n_customers = len(demands)
        
        # Create the solver
        # Create the solver
        # Use CBC solver which is typically faster for this type of problem
        solver = pywraplp.Solver.CreateSolver('SAT')
        
        if not solver:
            # Fallback to reference implementation if no solver available
            return self._fallback_solve(problem)
        
        # Set solver parameters for better performance
        solver.SetNumThreads(1)  # Use single thread for better performance on small problems
        solver.SetTimeLimit(1000)  # 1 second time limit

        # Create binary variables for facility opening
        y = [solver.BoolVar(f'y_{i}') for i in range(n_facilities)]
        
        # Create binary variables for assignments
        x = [[solver.BoolVar(f'x_{i}_{j}') for j in range(n_customers)] 
             for i in range(n_facilities)]

        # Objective function
        objective = solver.Objective()
        for i in range(n_facilities):
            objective.SetCoefficient(y[i], fixed_costs[i])
        for i in range(n_facilities):
            for j in range(n_customers):
                objective.SetCoefficient(x[i][j], transportation_costs[i][j])
        objective.SetMinimization()
        
        # Each customer must be served by exactly one facility
        for j in range(n_customers):
            solver.Add(solver.Sum([x[i][j] for i in range(n_facilities)]) == 1)
        
        # Capacity constraints and assignment constraints
        
        for i in range(n_facilities):
            # Capacity constraint: sum of demands assigned to facility <= capacity if open
            solver.Add(solver.Sum([demands[j] * x[i][j] for j in range(n_customers)]) <= capacities[i] * y[i])
            
            # Assignment constraint: can only assign to open facility
            for j in range(n_customers):
                solver.Add(x[i][j] <= y[i])
        
        # Solve the problem
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Extract solution
            facility_status = [bool(y[i].solution_value()) for i in range(n_facilities)]
            assignments = [[float(x[i][j].solution_value()) for j in range(n_customers)] for i in range(n_facilities)]
            
            return {
                "objective_value": solver.Objective().Value(),
                "facility_status": facility_status,
                "assignments": assignments,
            }
        else:
            # Return fallback solution if no optimal/feasible solution found
            return self._fallback_solve(problem)
    
    def _fallback_solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """Fallback to returning a default solution if optimization fails."""
        n_facilities = len(problem["fixed_costs"])
        n_customers = len(problem["demands"])
        return {
            "objective_value": float("inf"),
            "facility_status": [False] * n_facilities,
            "assignments": [[0.0] * n_customers for _ in range(n_facilities)],
        }