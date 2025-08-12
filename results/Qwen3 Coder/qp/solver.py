import numpy as np
from scipy import sparse
import osqp
from typing import Any, Dict

class Solver:
    def solve(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Solve convex quadratic optimization problem with highly optimized settings."""
        # Extract and convert problem data to appropriate types
        P = np.asarray(problem["P"], dtype=np.double)
        q = np.asarray(problem["q"], dtype=np.double)
        G = np.asarray(problem["G"], dtype=np.double)
        h = np.asarray(problem["h"], dtype=np.double)
        A = np.asarray(problem["A"], dtype=np.double)
        b = np.asarray(problem["b"], dtype=np.double)
        
        # Convert to sparse matrices for OSQP
        P_sparse = sparse.csc_matrix(P)
        A_eq_sparse = sparse.csc_matrix(A)
        G_sparse = sparse.csc_matrix(G)

        # Combine equality and inequality constraints
        # OSQP format: l <= Ax <= u
        A_combined = sparse.vstack([A_eq_sparse, G_sparse], format='csc')
        l = np.hstack([b, np.full(len(h), -np.inf)])
        u = np.hstack([b, h])

        # Create OSQP problem with aggressive settings
        prob = osqp.OSQP()
        prob.setup(P_sparse, q, A_combined, l, u, 
                  verbose=False, 
                  polish=True,
                  polish_refine_iter=1,
                  eps_abs=5e-5, 
                  eps_rel=5e-5, 
                  max_iter=200,
                  adaptive_rho=True,
                  check_termination=5)
        
        # First fast solve
        result = prob.solve()
        
        # If not accurate enough, refine
        if result.info.status == 'solved_inaccurate':
            prob.update_settings(eps_abs=1e-6, eps_rel=1e-6, max_iter=600)
            result = prob.solve()

        # Final check and fallback
        if result.info.status != 'solved' and result.info.status != 'solved_inaccurate':
            # Fallback to cvxpy if OSQP fails
            try:
                import cvxpy as cp
                n = len(q)
                x = cp.Variable(n)
                P_cp = cp.psd_wrap(P)
                objective_cp = 0.5 * cp.quad_form(x, P_cp) + q @ x
                constraints = [G @ x <= h, A @ x == b]
                prob_cp = cp.Problem(cp.Minimize(objective_cp), constraints)
                prob_cp.solve(solver=cp.OSQP, 
                             eps_abs=1e-6, 
                             eps_rel=1e-6, 
                             max_iter=500,
                             verbose=False)
                if prob_cp.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    return {"solution": x.value.tolist()}
            except:
                pass
            raise ValueError(f"Solver failed (status = {result.info.status})")

        return {"solution": result.x.tolist()}