from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
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
        c = problem["c"] if isinstance(problem["c"], np.ndarray) else np.array(problem["c"])
        b = problem["b"] if isinstance(problem["b"], np.ndarray) else np.array(problem["b"])
        P = problem["P"] if isinstance(problem["P"], np.ndarray) else np.array(problem["P"])
        q = problem["q"] if isinstance(problem["q"], np.ndarray) else np.array(problem["q"])
        m = len(P)
        n = len(c)
        
        x = cp.Variable(n)
        
        constraints = [None] * m
        for i in range(m):
            constraints[i] = cp.SOC(b[i] - q[i].T @ x, P[i].T @ x)
        
        problem_cp = cp.Problem(cp.Minimize(c.T @ x), constraints)
        
        try:
            problem_cp.solve(solver=cp.ECOS, verbose=False, warm_start=True, max_iters=100)
            
            # Check if a solution was found
            if problem_cp.status not in ["optimal", "optimal_inaccurate"]:
                return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}
            
            return {"objective_value": problem_cp.value, "x": x.value}
        
        except Exception as e:
            return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}