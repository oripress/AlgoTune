from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the lp centering problem using CVXPY.
        
        :param problem: A dictionary of the lp centering problem's parameters.
        :return: A dictionary with key:
                 "solution": a 1D list with n elements representing the solution to the lp centering problem.
        """
        # Convert inputs to numpy arrays for faster processing
        c = np.array(problem["c"], dtype=np.float64, order='C')
        A = np.array(problem["A"], dtype=np.float64, order='C')
        b = np.array(problem["b"], dtype=np.float64, order='C')
        n = c.shape[0]
        
        # Pre-allocate variable
        x = cp.Variable(n)
        
        # Formulate and solve problem
        objective = cp.Minimize(c.T @ x - cp.sum(cp.log(x)))
        constraints = [A @ x == b]
        prob = cp.Problem(objective, constraints)
        
        # Solve with optimized settings
        prob.solve(solver=cp.CLARABEL, verbose=False)
        
        # Return solution
        return {"solution": x.value.tolist()}