from typing import Any
import cvxpy as cp
import numpy as np

class Solver:
    def __init__(self):
        # Pre-configure solver settings for better performance
        self.solver_settings = {
            'max_iter': 200,
            'tol_gap_abs': 1e-7,
            'tol_gap_rel': 1e-7,
            'tol_feas': 1e-7,
            'tol_infeas_abs': 1e-7,
            'tol_infeas_rel': 1e-7,
        }
    
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
        A = np.asarray(problem["A"], dtype=np.float64)
        B = np.asarray(problem["B"], dtype=np.float64)
        n, m = A.shape[0], B.shape[1]
        
        # Define variables for the SDP
        Q = cp.Variable((n, n), symmetric=True)
        L = cp.Variable((m, n))
        
        # Pre-compute matrices for efficiency
        eye_n = np.eye(n, dtype=np.float64)
        eye_2n = np.eye(2 * n, dtype=np.float64)
        
        # Set up constraints - using a more efficient formulation
        constraints = [
            cp.bmat([[Q, Q @ A.T + L.T @ B.T], [A @ Q + B @ L, Q]]) >> eye_2n,
            Q >> eye_n,
        ]
        
        # Define objective
        objective = cp.Minimize(0)
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.CLARABEL, **self.solver_settings)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                Q_val = Q.value
                L_val = L.value
                
                # More efficient computation - avoid redundant inversions
                Q_inv = np.linalg.inv(Q_val)
                K_value = L_val @ Q_inv
                
                return {"is_stabilizable": True, "K": K_value.tolist(), "P": Q_inv.tolist()}
            else:
                return {"is_stabilizable": False, "K": None, "P": None}
                
        except Exception:
            return {"is_stabilizable": False, "K": None, "P": None}