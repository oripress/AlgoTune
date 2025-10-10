import cvxpy as cp
import numpy as np

class Solver:
    def solve(
        self, problem: dict[str, list[list[int]] | list[float] | int]
    ) -> dict[str, list[list[float]] | float]:
        """
        Solves the Perron-Frobenius matrix completion using CVXPY.

        Args:
            problem: Dict containing inds, a, n.

        Returns:
            Dict with estimates B, optimal_value.
        """
        inds = np.array(problem["inds"], dtype=np.int32)
        a = np.array(problem["a"], dtype=np.float64)
        n = problem["n"]

        # More efficient way to find unobserved indices
        observed_mask = np.zeros((n, n), dtype=bool)
        observed_mask[inds[:, 0], inds[:, 1]] = True
        otherinds = np.argwhere(~observed_mask)

        # --- Define CVXPY Variables ---
        B = cp.Variable((n, n), pos=True)

        # --- Define Objective ---
        objective = cp.Minimize(cp.pf_eigenvalue(B))

        # --- Define Constraints ---
        constraints = [
            cp.prod(B[otherinds[:, 0], otherinds[:, 1]]) == 1.0,
            B[inds[:, 0], inds[:, 1]] == a,
        ]

        # --- Solve Problem ---
        prob = cp.Problem(objective, constraints)
        
        # Try MOSEK first (fastest commercial solver for GPs)
        try:
            result = prob.solve(gp=True, solver=cp.MOSEK, verbose=False)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] and B.value is not None:
                return {
                    "B": B.value.tolist(),
                    "optimal_value": result,
                }
        except:
            pass
        
        # Fallback to default solver
        try:
            result = prob.solve(gp=True, verbose=False)
        except cp.SolverError as e:
            return None
        except Exception as e:
            return None

        # Check solver status
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return None

        if B.value is None:
            return None

        return {
            "B": B.value.tolist(),
            "optimal_value": result,
        }