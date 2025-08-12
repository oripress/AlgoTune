from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, np.ndarray]) -> dict[str, Any]:
        """
        Solves a given robust LP using CVXPY with optimized SOCP formulation.
        
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
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Convert to numpy arrays efficiently
        c = np.array(problem["c"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        P = [np.array(p_i, dtype=np.float64) for p_i in problem["P"]]
        q = [np.array(q_i, dtype=np.float64) for q_i in problem["q"]]
        m = len(P)
        n = len(c)
        
        x = cp.Variable(n)
        
        # Optimize constraint building by pre-computing matrix operations
        constraints = []
        for i in range(m):
            # Pre-compute the matrix-vector product for better performance
            Px_i = P[i].T @ x
            qx_i = q[i].T @ x
            constraints.append(cp.SOC(b[i] - qx_i, Px_i))
        
        problem = cp.Problem(cp.Minimize(c.T @ x), constraints)
        
        try:
            # Use ECOS with optimized parameters for best performance
            problem.solve(
                solver=cp.ECOS, 
                verbose=False,
                feastol=1e-8,
                reltol=1e-8,
                abstol=1e-8,
                max_iters=1000
            )
            
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}
            
            return {"objective_value": problem.value, "x": x.value}
            
        except Exception as e:
            return {"objective_value": float("inf"), "x": np.array([np.nan] * n)}