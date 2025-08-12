from typing import Any
import numpy as np
import osqp
from scipy.sparse import vstack, identity, csc_matrix

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list[float]]:
        """
        Solves the Markowitz portfolio optimization problem using the OSQP solver.
        """
        μ = np.asarray(problem["μ"], dtype=np.double)
        Σ = np.asarray(problem["Σ"], dtype=np.double)
        γ = float(problem["γ"])
        n = μ.size

        # OSQP minimizes: (1/2) * x'Px + q'x
        # Subject to:   l <= Ax <= u
        #
        # Our problem is to minimize: γ * w'Σw - μ'w
        #
        # By comparing the two forms, we get:
        # P = 2 * γ * Σ
        # q = -μ
        P = csc_matrix(2 * γ * Σ)
        q = -μ

        # Constraints:
        # 1. sum(w) = 1
        # 2. w >= 0
        # We combine these into the form l <= Aw <= u.
        
        # A_eq: sum(w) = 1
        A_eq = csc_matrix(np.ones((1, n)))
        
        # A_ineq: w >= 0  (equivalent to I*w >= 0)
        A_ineq = identity(n, format='csc')
        
        # Stack the constraint matrices
        A = vstack([A_eq, A_ineq], format='csc')

        # Define the lower and upper bounds for the constraints
        # For sum(w) = 1, lower bound is 1, upper bound is 1.
        # For w >= 0, lower bound is 0, upper bound is infinity.
        l = np.hstack([1.0, np.zeros(n)])
        u = np.hstack([1.0, np.full(n, np.inf)])

        # Create an OSQP object
        solver = osqp.OSQP()

        # Setup workspace and solve.
        # Settings are chosen to improve performance and robustness.
        solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False,
                     eps_abs=1e-08, eps_rel=1e-08,
                     max_iter=10000,
                     polish=True)
        
        results = solver.solve()

        # Check the solution status
        if results.info.status in ['solved', 'solved inaccurate']:
            return {"w": results.x.tolist()}

        return None