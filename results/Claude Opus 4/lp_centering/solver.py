import numpy as np
from typing import Any

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        """
        Solve the LP centering problem using Newton's method.
        
        minimize -c^T x + sum_i log(x_i)
        subject to Ax = b
        """
        c = np.array(problem["c"], dtype=np.float64)
        A = np.array(problem["A"], dtype=np.float64)
        b = np.array(problem["b"], dtype=np.float64)
        
        n = c.shape[0]
        m = A.shape[0]
        
        # Initialize with a strictly feasible point
        # First solve Ax = b to get a particular solution
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Ensure x > 0
        if np.any(x <= 0):
            # Shift x to be positive
            x = x - np.min(x) + 1.0
            
        # Newton's method
        for iteration in range(100):
            # Gradient: -c + 1/x
            grad = -c + 1.0 / x
            
            # Hessian: diag(1/x^2)
            H_diag = 1.0 / (x * x)
            
            # Form and solve KKT system
            # [H  A^T] [dx     ] = [-grad]
            # [A  0  ] [dlambda]   [0    ]
            
            # Using Schur complement:
            # S = A @ H^{-1} @ A^T
            # S * dlambda = A @ H^{-1} @ grad
            
            H_inv_diag = x * x  # H^{-1} = diag(x^2)
            
            # Compute Schur complement
            AH_inv = A * H_inv_diag[np.newaxis, :]  # A @ H^{-1}
            S = AH_inv @ A.T
            
            # Right hand side
            rhs = AH_inv @ grad
            
            # Solve for dual variable
            try:
                dlambda = np.linalg.solve(S, rhs)
            except np.linalg.LinAlgError:
                # Use least squares if singular
                dlambda = np.linalg.lstsq(S, rhs, rcond=None)[0]
            
            # Compute primal direction
            dx = -H_inv_diag * (grad + A.T @ dlambda)
            
            # Line search to maintain x > 0
            alpha = 1.0
            while np.any(x + alpha * dx <= 0) and alpha > 1e-16:
                alpha *= 0.5
            
            # Update
            x_new = x + alpha * dx
            
            # Check convergence
            if np.linalg.norm(dx) < 1e-10 * np.linalg.norm(x):
                break
                
            x = x_new
        
        return {"solution": x.tolist()}