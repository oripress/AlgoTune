import numpy as np
from typing import Any
from ortools.linear_solver import pywraplp

class Solver:
    def __init__(self):
        # Create a reusable solver instance
        self.solver = pywraplp.Solver.CreateSolver('GLOP')
        if not self.solver:
            self.solver = pywraplp.Solver.CreateSolver('CLP')
            
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the LP box problem using OR-Tools with GLOP solver.
        
        The problem is:
        minimize    c^T x
        subject to  Ax <= b
                   0 <= x <= 1
        """
        c = problem["c"]
        A = problem["A"]
        b = problem["b"]
        n = len(c)
        m = len(b)
        
        # Clear previous model
        self.solver.Clear()
        
        # Create variables with box constraints 0 <= x <= 1
        x = []
        for i in range(n):
            x.append(self.solver.NumVar(0, 1, f'x[{i}]'))
        
        # Add constraints Ax <= b
        for i in range(m):
            constraint = self.solver.Constraint(-self.solver.infinity(), b[i])
            for j in range(n):
                if A[i][j] != 0:
                    constraint.SetCoefficient(x[j], A[i][j])
        
        # Set objective: minimize c^T x
        objective = self.solver.Objective()
        for i in range(n):
            objective.SetCoefficient(x[i], c[i])
        objective.SetMinimization()
        
        # Solve
        status = self.solver.Solve()
        
        if status != pywraplp.Solver.OPTIMAL:
            raise ValueError(f"Optimization failed with status {status}")
        
        # Extract solution
        solution = [x[i].solution_value() for i in range(n)]
        
        return {"solution": solution}