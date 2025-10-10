from typing import Any
import numpy as np
import osqp
from scipy import sparse

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        P = np.asarray(problem["P"], float)
        q = np.asarray(problem["q"], float)
        G = np.asarray(problem["G"], float)
        h = np.asarray(problem["h"], float)
        A = np.asarray(problem["A"], float)
        b = np.asarray(problem["b"], float)
        
        # Symmetrize P
        P = (P + P.T) / 2
        
        # Convert to sparse matrices for OSQP
        P_sparse = sparse.csc_matrix(P)
        
        # Combine inequality and equality constraints
        # OSQP format: l <= Ax <= u
        A_combined = sparse.vstack([
            sparse.csc_matrix(G),
            sparse.csc_matrix(A)
        ], format='csc')
        
        l_combined = np.hstack([
            -np.inf * np.ones(len(h)),
            b
        ])
        
        u_combined = np.hstack([
            h,
            b
        ])
        
        # Setup OSQP
        solver = osqp.OSQP()
        solver.setup(
            P=P_sparse,
            q=q,
            A=A_combined,
            l=l_combined,
            u=u_combined,
            eps_abs=1e-8,
            eps_rel=1e-8,
            verbose=False,
            polish=False,  # Disable polish for speed
            adaptive_rho=True,  # Enable adaptive rho for faster convergence
            max_iter=4000
        )
        
        # Solve
        result = solver.solve()
        
        if result.info.status != 'solved':
            raise ValueError(f"Solver failed (status = {result.info.status})")
        
        objective_value = 0.5 * result.x @ P @ result.x + q @ result.x
        
        return {"solution": result.x.tolist(), "objective": float(objective_value)}