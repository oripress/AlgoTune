import numpy as np
from typing import Any
from scipy.linalg import solve_discrete_are
import scipy.linalg

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        """
        Solves the feedback controller design problem using LQR approach.
        
        Args:
            problem: A dictionary containing the system matrices A and B.
        
        Returns:
            A dictionary containing:
                - is_stabilizable: Boolean indicating if system is stabilizable
                - K: The feedback gain matrix (if stabilizable)
                - P: The Lyapunov matrix (if stabilizable)
        """
        A = np.ascontiguousarray(problem["A"], dtype=np.float64)
        B = np.ascontiguousarray(problem["B"], dtype=np.float64)
        n, m = A.shape[0], B.shape[1]
        
        try:
            # Create identity matrices once
            In = np.eye(n)
            Im = np.eye(m)
            
            # Solve discrete-time algebraic Riccati equation with Q=I, R=I
            P = solve_discrete_are(A, B, In, Im, balanced=False)
            
            # Compute LQR gain: K = (R + B'PB)^(-1) * B'PA
            BT = B.T
            BTP = BT @ P
            K = np.linalg.solve(BTP @ B + Im, BTP @ A)
            
            # Return -K since LQR gives u=-Kx but we need u=Kx
            return {"is_stabilizable": True, "K": (-K).tolist(), "P": P.tolist()}
                
        except (np.linalg.LinAlgError, ValueError):
            # If DARE fails, system is not stabilizable
            return {"is_stabilizable": False, "K": None, "P": None}