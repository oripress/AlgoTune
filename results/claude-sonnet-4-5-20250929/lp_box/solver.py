from typing import Any
import numpy as np
from ortools.linear_solver import pywraplp

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp box problem using OR-Tools.
        
        :param problem: A dictionary of the lp box problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution.
        """
        c = problem["c"]
        A = problem["A"]
        b = problem["b"]
        n = len(c)
        m = len(b)
        
        # Create solver
        solver = pywraplp.Solver.CreateSolver('GLOP')
        
        # Create variables with bounds [0, 1]
        x = [solver.NumVar(0.0, 1.0, '') for i in range(n)]
        
        # Add constraints A @ x <= b
        infinity = solver.infinity()
        for i in range(m):
            constraint = solver.Constraint(-infinity, b[i])
            row = A[i]
            for j in range(n):
                constraint.SetCoefficient(x[j], row[j])
        
        # Set objective
        objective = solver.Objective()
        for i in range(n):
            objective.SetCoefficient(x[i], c[i])
        objective.SetMinimization()
        
        # Solve
        solver.Solve()
        
        # Get solution
        solution = [x[i].solution_value() for i in range(n)]
        
        return {"solution": solution}