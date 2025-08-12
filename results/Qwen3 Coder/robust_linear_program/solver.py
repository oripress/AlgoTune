from typing import Any
import cvxpy as cp
import numpy as np
from numba import jit

class Solver:
    def solve(self, problem: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Solves a given robust LP using CVXPY.
        
        Args:
            problem: A dictionary with problem parameter:
                - c: vector defining linear objective of LP,
                - b: right-hand side scalars of linear constraint of LP,
                - P: list of m [n-by-n symmetric positive (semi-)definite matrices],
                - q: list of m [n-dimensional vectors]
        
        Returns:
            A dictionary containing the problem solution:
                - objective_value: the optimal objective value of robust LP,
                - x: the optimal solution.
        """
        # Extract problem data
        c = problem["c"]
        b = problem["b"]
        P = problem["P"]
        q = problem["q"]
        m = len(P)
        n = len(c)
        
        # Define optimization variable
        x = cp.Variable(n)
        
        # Define constraints more efficiently using vectorized operations
        # For each constraint: ||P_i^T * x||_2 <= b_i - q_i^T * x
        # This is the robust constraint with ellipsoidal uncertainty
        constraints = []
        for i in range(m):
            # Use cp.SOC (second-order cone) constraint for robust LP
            # SOC(t, x) means ||x||_2 <= t
            constraints.append(cp.SOC(b[i] - q[i] @ x, P[i].T @ x))
            
        # Define objective: minimize c^T * x  
        objective = cp.Minimize(c @ x)
        
        # Create and solve problem
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=False)
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                return {"objective_value": float("inf"), "x": np.array([float("nan")] * n)}
            
            # Return solution
            return {"objective_value": float(prob.value), "x": np.array(x.value)}
        except Exception:
            return {"objective_value": float("inf"), "x": np.array([float("nan")] * n)}