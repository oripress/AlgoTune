import cvxpy as cp
import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the optimal advertising problem using CVXPY.

        :param problem: Dictionary with problem parameters
        :return: Dictionary with optimal displays and revenue
        
        NOTE: Your solution must pass validation by:
        1. Returning correctly formatted output
        2. Having no NaN or infinity values
        3. Matching expected results within numerical tolerance
        """
        # Extract problem parameters
        P = np.array(problem["P"])
        R = np.array(problem["R"])
        B = np.array(problem["B"])
        c = np.array(problem["c"])
        T = np.array(problem["T"])

        # Derive m and n from P matrix
        m, n = P.shape

        # Define variables
        D = cp.Variable((m, n))

        # Define objective: maximize total revenue
        # Revenue for each ad is min(payment per click * total clicks, budget)
        clicks_per_ad = cp.sum(cp.multiply(P, D), axis=1)
        revenue_per_ad = cp.minimum(R * clicks_per_ad, B)
        total_revenue = cp.sum(revenue_per_ad)

        # Define constraints
        constraints = [
            D >= 0,  # Non-negative displays
            cp.sum(D, axis=0) <= T,  # Traffic capacity per time slot
            cp.sum(D, axis=1) >= c,  # Minimum display requirements
        ]

        # Define and solve the problem with optimized solver settings
        prob = cp.Problem(cp.Maximize(total_revenue), constraints)

        try:
            # Use ECOS solver (fastest for this problem)
            prob.solve(solver=cp.ECOS, verbose=False)

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return {"status": prob.status, "optimal": False}

            # Calculate actual revenue using vectorized operations
            D_val = D.value
            clicks = np.sum(P * D_val, axis=1)
            revenue = np.minimum(R * clicks, B)

            return {
                "status": prob.status,
                "optimal": True,
                "displays": D_val.tolist(),
                "clicks": clicks.tolist(),
                "revenue_per_ad": revenue.tolist(),
                "total_revenue": float(np.sum(revenue)),
                "objective_value": float(prob.value),
            }

        except cp.SolverError as e:
            return {"status": "solver_error", "optimal": False, "error": str(e)}
        except Exception as e:
            return {"status": "error", "optimal": False, "error": str(e)}