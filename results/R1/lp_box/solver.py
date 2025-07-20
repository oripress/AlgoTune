import os
os.environ['ORT_USE_PYPY'] = '1'  # Enable PyPy compatibility mode

from typing import Any
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, list]:
        c = problem["c"]
        A = problem["A"]
        b = problem["b"]
        n = len(c)
        m = len(b)
        
        # Create the linear solver with GLOP backend
        solver = pywraplp.Solver.CreateSolver('GLOP')
        if not solver:
            raise Exception("Could not create solver")
        
        # Create the variables without names for efficiency
        x = [solver.NumVar(0.0, 1.0, '') for _ in range(n)]
        
        # Set the objective
        objective = solver.Objective()
        for i in range(n):
            objective.SetCoefficient(x[i], c[i])
        objective.SetMinimization()
        
        # Add the constraints
        infinity = solver.infinity()
        # Use local variables to reduce attribute lookups in inner loops
        infinity = solver.infinity()
        # Use local variables to reduce attribute lookups in inner loops
        solver_Constraint = solver.Constraint
        for j in range(m):
            constraint = solver_Constraint(-infinity, b[j])
            set_coeff = constraint.SetCoefficient  # Cache method for this constraint
            row = A[j]  # Get the row once
            x_i = x  # Local reference to x
            for i in range(n):
                set_coeff(x_i[i], row[i])
        
        # Solve the problem
        status = solver.Solve()
        
        # Check the solution status
        if status != pywraplp.Solver.OPTIMAL:
            raise Exception("Problem could not be solved optimally")
        
        # Extract and return the solution
        solution = [x[i].solution_value() for i in range(n)]
        return {"solution": solution}