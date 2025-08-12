from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the feedback controller design problem using semidefinite programming.
        
        Args:
            problem: A dictionary containing the system matrices A and B.
            
        Returns:
            A dictionary containing:
                - is_stabilizable: Boolean indicating if the system is stabilizable
                - K: The feedback gain matrix
                - P: The Lyapunov matrix
        """
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        n, m = A.shape[0], B.shape[1]
        
        # Define variables for the SDP
        Q = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))
        constraints = [
            cp.bmat([[Q, Q @ A.T + L.T @ B.T], [A @ Q + B @ L, Q]]) >> np.eye(2 * n) * 0.01,
        ]
        objective = cp.Minimize(0)
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False, max_iter=200, tol_gap_abs=5e-2, tol_gap_rel=5e-2, tol_feas=5e-2)
            if prob.status in ["optimal", "optimal_inaccurate"]:
                K_value = L.value @ np.linalg.inv(Q.value)
                P = np.linalg.inv(Q.value)
                return {"is_stabilizable": True, "K": K_value.tolist(), "P": P.tolist()}
            else:
                return {"is_stabilizable": False, "K": None, "P": None}
        
        except Exception as e:
            return {"is_stabilizable": False, "K": None, "P": None}